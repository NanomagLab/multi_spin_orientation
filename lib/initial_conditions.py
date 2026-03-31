import tensorflow as tf
import numpy as np
from lib.magnetism import update_spin
from lib.spin_utils import (
    quat_mul,
    quat_inv,
    quat_pow,
    rotation_axis_and_angle_to_quat,
    get_positions,
    get_chebychev_boundary_mask,
    rotate_vector_by_quat,
    get_base_spin,
)


def quat_mul_np(q1: np.ndarray, q2: np.ndarray, dtype=np.float64) -> np.ndarray:
    out = quat_mul(
        tf.convert_to_tensor(np.asarray(q1), dtype=tf.as_dtype(dtype)),
        tf.convert_to_tensor(np.asarray(q2), dtype=tf.as_dtype(dtype)),
    )
    return np.asarray(out.numpy())


def quat_inv_np(q: np.ndarray, dtype=np.float64) -> np.ndarray:
    out = quat_inv(tf.convert_to_tensor(np.asarray(q), dtype=tf.as_dtype(dtype)))
    return np.asarray(out.numpy())


def quat_pow_np(q: np.ndarray, exponent: np.ndarray, dtype=np.float64) -> np.ndarray:
    out = quat_pow(
        tf.convert_to_tensor(np.asarray(q), dtype=tf.as_dtype(dtype)),
        tf.convert_to_tensor(np.asarray(exponent), dtype=tf.as_dtype(dtype)),
    )
    return np.asarray(out.numpy())


def rotation_axis_and_angle_to_quat_np(axis: np.ndarray, angle: np.ndarray, dtype=np.float64) -> np.ndarray:
    out = rotation_axis_and_angle_to_quat(
        tf.convert_to_tensor(np.asarray(axis), dtype=tf.as_dtype(dtype)),
        tf.convert_to_tensor(np.asarray(angle), dtype=tf.as_dtype(dtype)),
        dtype=tf.as_dtype(dtype),
    )
    return np.asarray(out.numpy())


def get_base_spin_np(multi_spin_mode: str, dtype=np.float64) -> np.ndarray:
    out = get_base_spin(multi_spin_mode=multi_spin_mode, dtype=tf.as_dtype(dtype))
    return np.asarray(out.numpy())


def get_positions_np(size: int, origin_on_grid: bool = False, dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = get_positions(size, origin_on_grid=origin_on_grid, dtype=tf.as_dtype(dtype))
    return np.asarray(yy.numpy()), np.asarray(xx.numpy())


def get_chebychev_boundary_mask_np(radius: int, origin_on_grid: bool, dtype=np.float64) -> np.ndarray:
    mask = get_chebychev_boundary_mask(radius, origin_on_grid=origin_on_grid, dtype=tf.as_dtype(dtype))
    return np.asarray(mask.numpy())


def rotate_vector_by_quat_np(v: np.ndarray, q: np.ndarray, dtype=np.float64) -> np.ndarray:
    out = rotate_vector_by_quat(
        tf.convert_to_tensor(np.asarray(v), dtype=tf.as_dtype(dtype)),
        tf.convert_to_tensor(np.asarray(q), dtype=tf.as_dtype(dtype)),
    )
    return np.asarray(out.numpy())


def get_radial_spin(
    size: int,
    axis: np.ndarray,
    angle: float,
    multi_spin_mode: str,
    origin_on_grid: bool = False,
    dtype: np.dtype = np.float64,
):
    yy, xx = get_positions_np(size, origin_on_grid=origin_on_grid, dtype=dtype)
    theta = np.arctan2(yy, xx)
    angle_multiplier = (theta % (2.0 * np.pi)) / (2.0 * np.pi)
    quat = rotation_axis_and_angle_to_quat_np(axis, angle * angle_multiplier, dtype=dtype)
    base_spin = get_base_spin_np(multi_spin_mode=multi_spin_mode, dtype=dtype)
    spins = rotate_vector_by_quat_np(base_spin, quat[..., np.newaxis, :])
    return spins.astype(dtype, copy=False)




def get_two_defects_spin_fields(
    quats_list: list[np.ndarray],
    defect_position: int,
    boundary_radius: int,
    multi_spin_mode: str,
    relaxation_step: int = 100,
    dtype: np.dtype = np.float64,
):
    def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Spherical linear interpolation between quaternions"""
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        t = np.asarray(t, dtype=float)
        
        q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
        q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
        
        dot = np.sum(q0 * q1, axis=-1, keepdims=True)
        q1 = np.where(dot < 0.0, -q1, q1)
        
        q_delta = quat_mul_np(quat_inv_np(q0), q1)
        if t.ndim > 0:
            q_delta_pow = quat_pow_np(q_delta[np.newaxis, :], t[:, np.newaxis])
            if q_delta_pow.ndim == 3:
                q_delta_pow = np.squeeze(q_delta_pow, axis=1)
            return quat_mul_np(q0[np.newaxis, :], q_delta_pow)
        else:
            return quat_mul_np(q0, quat_pow_np(q_delta, t))


    def _interpolate_row_segment(quat_field: np.ndarray, y_idx: int, x_start: int, x_end: int) -> np.ndarray:
        """Linear interpolation along a row of quaternions"""
        if x_end - x_start <= 1:
            return quat_field
        t = np.linspace(0.0, 1.0, x_end - x_start + 1, dtype=quat_field.dtype)[1:-1]
        q0 = quat_field[y_idx, x_start]
        q1 = quat_field[y_idx, x_end]
        quat_field[y_idx, x_start + 1:x_end] = _quat_slerp(q0, q1, t)
        return quat_field
        

    def _interpolate_col_segment(quat_field: np.ndarray, x_idx: int, y_start: int, y_end: int) -> np.ndarray:
        """Linear interpolation along a column of quaternions"""
        if y_end - y_start <= 1:
            return quat_field
        t = np.linspace(0.0, 1.0, y_end - y_start + 1, dtype=float)[1:-1]
        q0 = quat_field[y_start, x_idx]
        q1 = quat_field[y_end, x_idx]
        quat_field[y_start + 1:y_end, x_idx] = _quat_slerp(q0, q1, t)
        return quat_field


    def _get_two_defects_quat_field(
        defect_position: int,
        boundary_radius: int,
        quats: list[np.ndarray],
        dtype: np.dtype = np.float64,
    ):
        """Generate quaternion field including two defects and boundary for triangular lattice"""
        boundary_quat, defect_1_quat, defect_2_quat = quats
        boundary_mask = get_chebychev_boundary_mask_np(boundary_radius, origin_on_grid=True, dtype=dtype)
        size = boundary_mask.shape[0]
        yy, xx = get_positions_np(size, origin_on_grid=True, dtype=dtype)
        defect_mask = np.zeros_like(boundary_mask)
        defect_mask[size // 2 - 1, size // 2 + defect_position: size // 2 + defect_position + 2] = 1.
        defect_mask[size // 2, size // 2 + defect_position - 1: size // 2 + defect_position + 2] = 1.
        defect_mask[size // 2 + 1, size // 2 + defect_position - 1: size // 2 + defect_position + 1] = 1.
        defect_mask[size // 2 - 1, size // 2 - defect_position: size // 2 - defect_position + 2] = 1.
        defect_mask[size // 2, size // 2 - defect_position - 1: size // 2 - defect_position + 2] = 1.
        defect_mask[size // 2 + 1, size // 2 - defect_position - 1: size // 2 - defect_position + 1] = 1.

        # Background: azimuthal rotation
        theta = np.arctan2(yy, xx)
        exponent = theta % (2.0 * np.pi) / (2.0 * np.pi)
        quat_field = np.array(quat_pow_np(boundary_quat, exponent))

        # First defect (right side)
        center_x, center_y = size // 2 + defect_position, size // 2
        defect_1_bg = boundary_quat
        # right
        quat_field[center_y, center_x + 1] = quat_pow_np(defect_1_quat, 0.)
        # right upper
        quat_field[center_y + 1, center_x] = quat_pow_np(defect_1_quat, 1. / 6.)
        # left upper
        quat_field[center_y + 1, center_x - 1] = quat_pow_np(defect_1_quat, 2. / 6.)
        # left
        quat_field[center_y, center_x - 1] = quat_pow_np(defect_1_quat, 3. / 6.)
        # left lower
        quat_field[center_y - 1, center_x] = quat_mul_np(quat_pow_np(defect_1_quat, -2. / 6.), defect_1_bg)
        # right lower
        quat_field[center_y - 1, center_x + 1] = quat_mul_np(quat_pow_np(defect_1_quat, -1. / 6.), defect_1_bg)

        # Second defect (left side)
        center_x, center_y = size // 2 - defect_position, size // 2
        defect_2_bg = quat_pow_np(boundary_quat, 0.5)
        # right
        quat_field[center_y, center_x + 1] = quat_mul_np(defect_2_bg, quat_pow_np(defect_2_quat, -3. / 6.))
        # right upper
        quat_field[center_y + 1, center_x] = quat_mul_np(defect_2_bg, quat_pow_np(defect_2_quat, -2. / 6.))
        # left upper
        quat_field[center_y + 1, center_x - 1] = quat_mul_np(defect_2_bg, quat_pow_np(defect_2_quat, -1. / 6.))
        # left
        quat_field[center_y, center_x - 1] = quat_mul_np(defect_2_bg, quat_pow_np(defect_2_quat, 0.))
        # left lower
        quat_field[center_y - 1, center_x] = quat_mul_np(defect_2_bg, quat_pow_np(defect_2_quat, 1. / 6.))
        # right lower
        quat_field[center_y - 1, center_x + 1] = quat_mul_np(defect_2_bg, quat_pow_np(defect_2_quat, 2. / 6.))

        defect_y = size // 2
        defect1_x = size // 2 + defect_position
        defect2_x = size // 2 - defect_position

        # Horizontal interpolation
        # top
        x_boundary_end = 0
        for i in range(size):
            if boundary_mask[defect_y + 1, i] < 0.5:
                break
            x_boundary_end = i
        x_boundary_start = size
        for i in range(size - 1, -1, -1):
            if boundary_mask[defect_y + 1, i] < 0.5:
                break
            x_boundary_start = i
        quat_field = _interpolate_row_segment(quat_field, defect_y + 1, x_boundary_end, defect2_x - 1)
        quat_field = _interpolate_row_segment(quat_field, defect_y + 1, defect2_x, defect1_x - 1)
        quat_field = _interpolate_row_segment(quat_field, defect_y + 1, defect1_x, x_boundary_start)
        # middel
        x_boundary_end = 0
        for i in range(size):
            if boundary_mask[defect_y, i] < 0.5:
                break
            x_boundary_end = i
        x_boundary_start = size
        for i in range(size - 1, -1, -1):
            if boundary_mask[defect_y, i] < 0.5:
                break
            x_boundary_start = i
        quat_field = _interpolate_row_segment(quat_field, defect_y, x_boundary_end, defect2_x - 1)
        quat_field = _interpolate_row_segment(quat_field, defect_y, defect2_x + 1, defect1_x - 1)
        quat_field = _interpolate_row_segment(quat_field, defect_y, defect1_x + 1, x_boundary_start)
        # bottom
        x_boundary_end = 0
        for i in range(size):
            if boundary_mask[defect_y - 1, i] < 0.5:
                break
            x_boundary_end = i
        x_boundary_start = size - 1
        for i in range(size - 1, -1, -1):
            if boundary_mask[defect_y - 1, i] < 0.5:
                break
        quat_field = _interpolate_row_segment(quat_field, defect_y - 1, x_boundary_end, defect2_x)
        quat_field = _interpolate_row_segment(quat_field, defect_y - 1, defect2_x + 1, defect1_x)
        quat_field = _interpolate_row_segment(quat_field, defect_y - 1, defect1_x + 1, x_boundary_start)

        for x_idx in range(size):
            y_boundary_end = 0
            for i in range(size):
                if boundary_mask[i, x_idx] < 0.5:
                    break
                y_boundary_end = i
            y_boundary_start = size
            for i in range(size - 1, -1, -1):
                if boundary_mask[i, x_idx] < 0.5:
                    break
                y_boundary_start = i
            if y_boundary_end > y_boundary_start:
                continue
            quat_field = _interpolate_col_segment(quat_field, x_idx, y_boundary_end, defect_y - 1)
            quat_field = _interpolate_col_segment(quat_field, x_idx, defect_y + 1, y_boundary_start)
        return quat_field, boundary_mask, defect_mask


    quat_fields = []
    for quats in quats_list:
        quat_field, boundary_mask, defect_mask = _get_two_defects_quat_field(
            defect_position=defect_position,
            boundary_radius=boundary_radius,
            quats=quats,
            dtype=dtype,
        )
        quat_fields.append(quat_field)
    fix_mask = boundary_mask + defect_mask
    quat_fields = np.stack(quat_fields, axis=0)
    base_spin_np = get_base_spin_np(multi_spin_mode=multi_spin_mode, dtype=dtype)
    spins_np = rotate_vector_by_quat_np(base_spin_np, quat_fields[..., np.newaxis, :]).astype(dtype, copy=False)

    spins = tf.convert_to_tensor(spins_np, dtype=tf.as_dtype(dtype))
    fix_mask_tf = tf.convert_to_tensor(fix_mask.astype(dtype, copy=False), dtype=tf.as_dtype(dtype))
    for _ in range(relaxation_step):
        spins = update_spin(spins, multi_spin_mode=multi_spin_mode, fix_mask=fix_mask_tf)
    boundary_mask_tf = tf.convert_to_tensor(boundary_mask.astype(dtype, copy=False), dtype=tf.as_dtype(dtype))
    return spins, boundary_mask_tf



def get_random_spin_fields(
    radius: int,
    multi_spin_mode: str,
    dtype: tf.DType = tf.float64,
):
    """
    Random bulk spins with fixed boundary, same as the original TensorFlow path:
    uses ``tf.random.normal`` so results match ``tf.random.set_seed(...)`` (not ``np.random``).
    Returns a NumPy array for ``np.save`` / plotting.
    """
    tf_dtype = dtype if isinstance(dtype, tf.DType) else tf.as_dtype(dtype)
    boundary_mask = get_chebychev_boundary_mask(radius, origin_on_grid=True, dtype=tf_dtype)
    size = int(boundary_mask.shape[0])
    base_spin = get_base_spin(multi_spin_mode=multi_spin_mode, dtype=tf_dtype)
    boundary_spin = tf.broadcast_to(base_spin, (size, size) + tuple(base_spin.shape))
    random_quats = tf.random.normal((size, size, 4), dtype=tf_dtype)
    random_quats = tf.math.l2_normalize(random_quats, axis=-1)
    random_spins = rotate_vector_by_quat(
        base_spin,
        random_quats[..., tf.newaxis, :],
    )
    init_spins = (
        random_spins * (1.0 - boundary_mask[..., tf.newaxis, tf.newaxis])
        + boundary_spin * boundary_mask[..., tf.newaxis, tf.newaxis]
    )
    return np.asarray(init_spins.numpy())
