from __future__ import annotations

import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm


def _resolve_couplings(
    multi_spin_mode: str,
    JA1: float,
    JA2: float,
    JF1: float,
    JF2: float,
    JF3: float,
    JF4: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Return (JA1, JA2, JF1, JF2, JF3, JF4) with mode-specific defaults applied.

    multi_spin_mode: "three_spin", "four_spin" or "custom"
    """
    if multi_spin_mode == "three_spin":
        return -10.0, 0.0, 0.0, 5.0 / 9.0, 4.0 / 9.0, 0.0
    elif multi_spin_mode == "four_spin":
        return 0.0, -20.0, 0.0, 0.0, 1.0, 0.0
    elif multi_spin_mode == "custom":
        return JA1, JA2, JF1, JF2, JF3, JF4
    else:
        raise ValueError(f"Invalid multi_spin_mode: {multi_spin_mode}")


@tf.function
def get_heff_multi_spin(
        spins: tf.Tensor|np.ndarray,
        multi_spin_mode: str,
        JA1: float = -0.,
        JA2: float = -0.,
        JF1: float = 0.,
        JF2: float = 0.,
        JF3: float = 0.,
        JF4: float = 0.,
) -> tf.Tensor:
    """
    Calculate effective magnetic field for multi-spin systems.
    
    :param spins: (..., Height, Width, n_spins, 3) - arbitrary batch dimensions followed by spatial and spin dimensions
    :param JA1 ~ JA2: Antiferromagnetic coupling of corresponding polynomial order
                      (interactions within the same pixel)
    :param JF1 ~ JF4: Ferromagnetic coupling of corresponding polynomial order
                      (interactions between different pixels)
    :return: heff: (..., Height, Width, n_spins, 3) - effective field with same shape as input
    """

    JA1, JA2, JF1, JF2, JF3, JF4 = _resolve_couplings(
        multi_spin_mode, JA1, JA2, JF1, JF2, JF3, JF4
    )
    Si = spins
    n_spins = tf.shape(spins)[-2]

    # Self-interaction term: adjusts update speed during dynamics
    J_self = 2. * float(n_spins) * (abs(JA1) + abs(JA2))
    J_self = tf.cast(J_self, dtype=spins.dtype)

    # Antiferromagnetic interactions (within same pixel - multiple spins per pixel)
    Si_hetero_shape = tf.concat([tf.shape(Si)[:-1], [n_spins, 3]], axis=0)
    Si_hetero_mask = tf.broadcast_to(1. - tf.eye(n_spins, dtype=spins.dtype)[..., tf.newaxis], Si_hetero_shape)
    Si_hetero = tf.broadcast_to(Si[..., tf.newaxis, :, :], Si_hetero_shape) * Si_hetero_mask
    Si_dot_Si_hetero = tf.reduce_sum(Si[..., tf.newaxis, :] * Si_hetero, axis=-1, keepdims=True)
    heff_A1 = JA1 * tf.reduce_sum(Si_hetero, axis=-2)
    heff_A2 = JA2 * 2. * tf.reduce_sum(Si_dot_Si_hetero * Si_hetero, axis=-2)
    heff_A = heff_A1 + heff_A2

    right_top = tf.roll(spins, -1, axis=-4)
    left_bottom = tf.roll(spins, 1, axis=-4)
    left = tf.roll(spins, 1, axis=-3)
    right = tf.roll(spins, -1, axis=-3)
    lefttop = tf.roll(right_top, 1, axis=-3)
    rightbottom = tf.roll(left_bottom, -1, axis=-3)
    Sj = tf.stack([right_top, left_bottom, left, right, lefttop, rightbottom], axis=-2)
    Sj = Sj[..., tf.newaxis, :, :, :]
    Si_dot_Sj = tf.reduce_sum(
        Si[..., tf.newaxis, tf.newaxis, :] * Sj,
        axis=-1,
        keepdims=True
    )

    heff_F1 = JF1 * 0.5 * tf.reduce_sum(Sj, axis=(-2, -3))
    heff_F2 = JF2 * 1.0 * tf.reduce_sum((Si_dot_Sj * Sj), axis=(-2, -3))
    heff_F3 = JF3 * 1.5 * tf.reduce_sum((Si_dot_Sj ** 2 * Sj), axis=(-2, -3))
    heff_F4 = JF4 * 2.0 * tf.reduce_sum((Si_dot_Sj ** 3 * Sj), axis=(-2, -3))
    heff_F = heff_F1 + heff_F2 + heff_F3 + heff_F4

    # Self-energy term
    heff_self = J_self * spins

    heff = heff_A + heff_F + heff_self
    return heff


@tf.function
def get_energy_density_multi_spin(
        spins,
        multi_spin_mode: str,
        JA1: float = -0.,
        JA2: float = -0.,
        JF1: float = 0.,
        JF2: float = 0.,
        JF3: float = 0.,
        JF4: float = 0.,
) -> tf.Tensor:
    """
    Calculate energy density for multi-spin systems.

    :param spins: (..., Height, Width, n_spins, 3) - arbitrary batch dimensions followed by spatial and spin dimensions
    :param JA1 ~ JA2: Antiferromagnetic coupling of corresponding polynomial order
                      (interactions within the same pixel)
    :param JF1 ~ JF4: Ferromagnetic coupling of corresponding polynomial order
                      (interactions between different pixels)
    :return: E_density: (..., Height, Width) - energy density per pixel
    """
    
    JA1, JA2, JF1, JF2, JF3, JF4 = _resolve_couplings(
        multi_spin_mode, JA1, JA2, JF1, JF2, JF3, JF4
    )

    Si = spins
    n_spins = tf.shape(spins)[-2]

    # Antiferromagnetic energy (within same pixel)
    Si_hetero_shape = tf.concat([tf.shape(Si)[:-1], [n_spins, 3]], axis=0)
    Si_hetero_mask = tf.broadcast_to(1. - tf.eye(n_spins, dtype=spins.dtype)[..., tf.newaxis], Si_hetero_shape)
    Si_hetero = tf.broadcast_to(Si[..., tf.newaxis, :, :], Si_hetero_shape) * Si_hetero_mask
    Si_dot_Si_hetero = tf.reduce_sum(Si[..., tf.newaxis, :] * Si_hetero, axis=-1)
    E_A1 = -JA1 * tf.reduce_sum(Si_dot_Si_hetero, axis=(-1, -2))
    E_A2 = -JA2 * tf.reduce_sum(Si_dot_Si_hetero ** 2, axis=(-1, -2))
    E_A = E_A1 + E_A2

    right_top = tf.roll(spins, -1, axis=-4)
    left_bottom = tf.roll(spins, 1, axis=-4)
    left = tf.roll(spins, 1, axis=-3)
    right = tf.roll(spins, -1, axis=-3)
    lefttop = tf.roll(right_top, 1, axis=-3)
    rightbottom = tf.roll(left_bottom, -1, axis=-3)
    Sj = tf.stack([right_top, left_bottom, left, right, lefttop, rightbottom], axis=-2)
    Sj = Sj[..., tf.newaxis, :, :, :]

    Si_dot_Sj = tf.reduce_sum(
        Si[..., tf.newaxis, tf.newaxis, :] * Sj,
        axis=-1
    )

    E_F1 = -JF1 * 0.5 * tf.reduce_sum(Si_dot_Sj, axis=(-1, -2, -3))
    E_F2 = -JF2 * 0.5 * tf.reduce_sum(Si_dot_Sj ** 2, axis=(-1, -2, -3))
    E_F3 = -JF3 * 0.5 * tf.reduce_sum(Si_dot_Sj ** 3, axis=(-1, -2, -3))
    E_F4 = -JF4 * 0.5 * tf.reduce_sum(Si_dot_Sj ** 4, axis=(-1, -2, -3))
    E_F = E_F1 + E_F2 + E_F3 + E_F4
    
    return E_A + E_F


def update_spin(
    spins: tf.Tensor|np.ndarray,
    multi_spin_mode: str,
    JA1: float = -0.,
    JA2: float = -0.,
    JF1: float = 0.,
    JF2: float = 0.,
    JF3: float = 0.,
    JF4: float = 0.,
    fix_mask: np.ndarray = None,
) -> tf.Tensor:
    """
    Update the spins.
    :param spins: (..., Height, Width, n_spins, 3) - arbitrary batch dimensions followed by spatial and spin dimensions
    :param multi_spin_mode: "three_spin", "four_spin" or "custom"
    :param lattice_mode: "square" or "triangular"
    :param JA1 ~ JA2: Antiferromagnetic coupling of corresponding polynomial order
    :param JF1 ~ JF4: Ferromagnetic coupling of corresponding polynomial order
    :param fix_mask: (..., Height, Width, 1) - mask of the spins to fix
    :return: spins: (..., Height, Width, n_spins, 3) - updated spins
    """
    heff = get_heff_multi_spin(
        spins,
        multi_spin_mode=multi_spin_mode,
        JA1=JA1,
        JA2=JA2,
        JF1=JF1,
        JF2=JF2,
        JF3=JF3,
        JF4=JF4,
    )
    fix_mask = fix_mask[..., tf.newaxis, tf.newaxis]
    updated_spins = tf.math.l2_normalize(heff, axis=-1)
    if fix_mask is not None:
        updated_spins = updated_spins * (1. - fix_mask) + spins * fix_mask
    return updated_spins
