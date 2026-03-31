from __future__ import annotations

import tensorflow as tf
import numpy as np


def quat_mul(q1: tf.Tensor, q2: tf.Tensor) -> tf.Tensor:
    """
    Multiply two quaternions with automatic broadcasting.
    
    :param q1: (4,) or (..., 4) - first quaternion [w, x, y, z]
    :param q2: (4,) or (..., 4) - second quaternion [w, x, y, z]
    :return: (..., 4) - product quaternion [w, x, y, z]
    """
    q1 = tf.math.l2_normalize(q1, axis=-1)
    q2 = tf.math.l2_normalize(q2, axis=-1)

    # Broadcast shapes
    shape = tf.broadcast_dynamic_shape(tf.shape(q1)[:-1], tf.shape(q2)[:-1])
    shape = tf.concat([shape, [4]], axis=0)
    q1 = tf.broadcast_to(q1, shape)
    q2 = tf.broadcast_to(q2, shape)

    w1, x1, y1, z1 = tf.unstack(q1, axis=-1)
    w2, x2, y2, z2 = tf.unstack(q2, axis=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return tf.stack([w, x, y, z], axis=-1)


def quat_inv(q: tf.Tensor) -> tf.Tensor:
    """
    Compute quaternion inverse (conjugate).
    
    :param q: (..., 4) - quaternion [w, x, y, z]
    :return: (..., 4) - inverse quaternion
    """
    w, x, y, z = tf.unstack(q, axis=-1)
    return tf.stack([w, -x, -y, -z], axis=-1)


def quat_pow(q: tf.Tensor, exponent: tf.Tensor) -> tf.Tensor:
    """
    Raise quaternion to a power with automatic broadcasting.
    
    :param q: (4,) or (..., 4) - quaternion [w, x, y, z]
    :param exponent: (...,) or scalar - exponent values
    :return: (..., 4) - quaternion raised to power
    """
    # Cast exponent to match quaternion dtype
    exponent = tf.cast(exponent, q.dtype)
    
    # Get batch shapes
    q_batch_shape = tf.shape(q)[:-1]
    exp_shape = tf.shape(exponent)
    
    # Dynamically broadcast batch shapes
    broadcast_shape = tf.broadcast_dynamic_shape(q_batch_shape, exp_shape)
    
    # Broadcast q and exponent to the common shape
    q = tf.broadcast_to(q, tf.concat([broadcast_shape, [4]], axis=0))
    exponent = tf.broadcast_to(exponent, broadcast_shape)
    exponent = exponent[..., tf.newaxis]
    
    w = q[..., :1]
    v = q[..., 1:]

    # Calculate angle and axis
    theta = 2.0 * tf.acos(tf.clip_by_value(w, -1.0, 1.0))
    sin_half_theta = tf.sqrt(1.0 - w ** 2)

    # Rotation axis (prevent division by zero)
    axis = tf.where(
        sin_half_theta > 1e-6,
        v / sin_half_theta,
        tf.zeros_like(v)
    )

    # Calculate new rotation
    new_theta = theta * exponent
    new_w = tf.cos(new_theta / 2.0)
    new_vec = axis * tf.sin(new_theta / 2.0)

    result = tf.concat([new_w, new_vec], axis=-1)
    return result


def rotation_axis_and_angle_to_quat(axis: tf.Tensor, angle: tf.Tensor, dtype: tf.DType = tf.float64) -> tf.Tensor:
    """
    Convert axis-angle representation to quaternion.
    
    :param axis: (..., 3) - rotation axis (unit vector)
    :param angle: (...,) or scalar - rotation angle in radians
    :return: (..., 4) - unit quaternion [w, x, y, z]
    """
    # Normalize axis
    axis = tf.math.l2_normalize(axis, axis=-1)
    axis = tf.cast(axis, dtype=dtype)
    angle = tf.cast(angle, dtype=dtype)
    
    # Broadcast shapes
    axis_batch_shape = tf.shape(axis)[:-1]
    angle_shape = tf.shape(angle)
    broadcast_shape = tf.broadcast_dynamic_shape(axis_batch_shape, angle_shape)
    
    axis = tf.broadcast_to(axis, tf.concat([broadcast_shape, [3]], axis=0))
    angle = tf.broadcast_to(angle, broadcast_shape)
    
    # Quaternion from axis-angle: q = [cos(θ/2), sin(θ/2) * axis]
    half_angle = angle / 2.0
    w = tf.cos(half_angle)
    xyz = tf.sin(half_angle)[..., tf.newaxis] * axis
    
    q = tf.concat([w[..., tf.newaxis], xyz], axis=-1)
    
    # Ensure w >= 0 for canonical form
    q = tf.where(q[..., :1] < 0.0, -q, q)
    
    return q


def get_base_spin(multi_spin_mode: str, dtype=tf.float64) -> tf.Tensor:
    if multi_spin_mode == "three_spin":
        base_spin = tf.constant([
            [1., 0., 0.],
            [-0.5, np.sqrt(3.) / 2., 0.],
            [-0.5, -np.sqrt(3.) / 2., 0.],
        ], dtype=dtype)
        return base_spin
    elif multi_spin_mode == "four_spin":
        # sqrt2 = np.sqrt(2.)
        # sqrt6 = np.sqrt(6.)
        # base_spin = np.array([
        #     [0., 0., 1.],                    # z unit vector
        #     [2. * sqrt2 / 3., 0., -1. / 3.], # y component = 0
        #     [-sqrt2 / 3., sqrt6 / 3., -1. / 3.],  # 120° rotation
        #     [-sqrt2 / 3., -sqrt6 / 3., -1. / 3.], # 240° rotation
        # ], dtype=dtype)
        base_spin = tf.constant([
            [1., 1., 1.],
            [1., -1., -1.],
            [-1., 1., -1.],
            [-1., -1., 1.],
        ], dtype=dtype) / tf.constant(np.sqrt(3.), dtype=dtype)
        return tf.constant(base_spin, dtype=dtype)
    else:
        raise NotImplementedError(f"Only multi_spin_mode == 'three_spin' or 'four_spin' is implemented")


def get_group(multi_spin_mode: str, dtype=tf.float64) -> tf.Tensor:
    if multi_spin_mode == "three_spin":
        r_quat = tf.constant([0.5, 0., 0., np.sqrt(3.) / 2.], dtype=dtype)
        s_quat = tf.constant([0., 1., 0., 0.], dtype=dtype)
        r_powers = tf.stack([quat_pow(r_quat, i) for i in range(6)])
        double_d3 = tf.concat([
            r_powers,
            quat_mul(s_quat, r_powers),
        ], axis=0)
        return double_d3
    elif multi_spin_mode == "four_spin":
        t_quat = tf.constant([0.5, 0.5, 0.5, 0.5], dtype=dtype)
        s_quat = tf.constant([0., 1., 0., 0.], dtype=dtype)
        t_powers = tf.stack([quat_pow(t_quat, i) for i in range(6)])
        double_a4 = tf.concat([
            t_powers,
            quat_mul(s_quat, t_powers),
            quat_mul(quat_mul(t_quat, s_quat), t_powers),
            quat_mul(quat_mul(quat_pow(t_quat, 2), s_quat), t_powers),
        ], axis=0)
        return double_a4
    else:
        raise NotImplementedError(f"Only multi_spin_model == 'three_spin' or 'four_spin' is implemented")


def get_conjugation_table(multi_spin_mode: str, dtype=tf.int32) -> tf.Tensor:
    if multi_spin_mode == "three_spin":
        # table = tf.constant([0, 1, 4, 5, 4, 1, 2, 3, 2, 3, 2, 3], dtype=dtype)
        table = tf.constant([0, 1, 2, 3, 2, 1, 4, 5, 4, 5, 4, 5], dtype=dtype)
        return table
    elif multi_spin_mode == "four_spin":
        # table = tf.constant([
        #     0, 1,
        #     2, 2, 2, 2, 2, 2,
        #     3, 4, 4, 3, 4, 3, 3, 4,
        #     5, 6, 6, 5, 6, 5, 5, 6
        # ], dtype=dtype) # For quaternion enumeration.
        # table = tf.constant([
        #     0, 1, 4, 6, 5, 2,
        #     3, 5, 4, 3, 1, 2,
        #     5, 4, 3, 1, 2, 3,
        #     4, 3, 1, 2, 3, 5
        # ], dtype=dtype) # For cosets of <t> enumeration
        table = tf.constant([
            0, 1, 2, 3, 4, 5,
            6, 4, 2, 6, 1, 5,
            4, 2, 6, 1, 5, 6,
            2, 6, 1, 5, 6, 4
        ], dtype=dtype) # For cosets of <t> enumeration
        return table
    else:
        raise NotImplementedError(f"Only multi_spin_mode == 'three_spin' or 'four_spin' is implemented")


def get_composition_rule(multi_spin_mode: str, dtype=tf.int32):
    group = get_group(multi_spin_mode=multi_spin_mode)
    g1_g2 = quat_mul(group[:, tf.newaxis], group[tf.newaxis]) # (n_groups, n_groups, 4)
    similarity = tf.reduce_sum(g1_g2[:, :, tf.newaxis] * group[tf.newaxis, tf.newaxis, :], axis=-1) # (n_groups, n_groups, n_groups)
    group_operation = tf.argmax(similarity, axis=-1) # (n_groups, n_groups)
    conjugation_table = get_conjugation_table(multi_spin_mode=multi_spin_mode)
    n_types = tf.reduce_max(conjugation_table) + 1
    composition_rule = [[] for _ in range(n_types)]
    for i in range(1, len(group)):
        type_i = conjugation_table[i]
        for j in range(1, len(group)):
            type_j = conjugation_table[j]
            gi_gj = group_operation[i, j]
            type_ij = conjugation_table[gi_gj]
            composition_rule[type_ij].append([int(type_i), int(type_j)])

    # Drop duplicate pairs: [a,b] == [b,a], and exact duplicates within the list
    def _unordered_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a <= b else (b, a)

    for k in range(len(composition_rule)):
        seen = set()
        uniq = []
        for pair in composition_rule[k]:
            a, b = int(pair[0]), int(pair[1])
            key = _unordered_key(a, b)
            if key not in seen:
                seen.add(key)
                uniq.append([key[0], key[1]])
        composition_rule[k] = uniq

    return composition_rule


def get_positions(size: int, origin_on_grid: bool = False, dtype: tf.DType = tf.float64) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Get positions of a triangular lattice.
    :param size: size of the lattice
    :param origin_on_grid: whether to place the origin on the grid
    :param dtype: dtype of the positions
    :return: (yy, xx) - positions of the lattice
    """
    ii, jj = tf.meshgrid(
        tf.range(size, dtype=dtype) - tf.cast(size // 2, dtype=dtype),
        tf.range(size, dtype=dtype) - tf.cast(size // 2, dtype=dtype),
        indexing='ij'
    )
    if not origin_on_grid:
        ii += 0.5
        jj += 0.5
    sqrt3_over_2 = tf.constant(np.sqrt(3.0) / 2.0, dtype=dtype)
    half = tf.constant(0.5, dtype=dtype)
    yy = sqrt3_over_2 * ii
    xx = jj + half * ii
    return yy, xx


def get_chebychev_boundary_mask(
    radius: int,
    origin_on_grid: bool,
    dtype: tf.DType = tf.float64,
):
    if origin_on_grid:
        size = radius * 2 + 3
    else:
        size = radius * 2 + 2
    ii, jj = tf.meshgrid(
        tf.range(size, dtype=dtype) - tf.cast(size // 2, dtype=dtype),
        tf.range(size, dtype=dtype) - tf.cast(size // 2, dtype=dtype),
        indexing='ij'
    )
    if not origin_on_grid:
        ii += 0.5
        jj += 0.5
    yy, xx = ii, jj

    distances = tf.stack([tf.abs(yy), tf.abs(xx), tf.abs(yy + xx)], axis=0)
    distance = tf.reduce_max(distances, axis=0)
    mask = tf.where(distance > radius, 1., 0.)
    return tf.cast(mask, dtype=dtype)



def rotate_vector_by_quat(v: tf.Tensor, q: tf.Tensor) -> tf.Tensor:
    """
    Rotate vector v by quaternion q.
    
    :param v: (..., 3) - vector to rotate
    :param q: (..., 4) - rotation quaternion [w, x, y, z]
    :return: (..., 3) - rotated vector
    """
    # Broadcast v and q to a common batch shape
    v_batch = tf.shape(v)[:-1]
    q_batch = tf.shape(q)[:-1]
    batch_shape = tf.broadcast_dynamic_shape(v_batch, q_batch)

    v = tf.broadcast_to(v, tf.concat([batch_shape, [3]], axis=0))
    q = tf.broadcast_to(q, tf.concat([batch_shape, [4]], axis=0))

    w, x, y, z = tf.unstack(q, axis=-1)
    q_vec = tf.stack([x, y, z], axis=-1)

    # Rotate vector using quaternion formula
    v_rot = v + 2.0 * tf.linalg.cross(q_vec, tf.linalg.cross(q_vec, v) + w[..., tf.newaxis] * v)
    return v_rot

def _get_identical_quaternions(q: tf.Tensor, symmetry: tf.Tensor) -> tf.Tensor:
    """
    Return all quaternions equivalent to q under the given symmetry group.
    
    Generates all g ∈ symmetry such that q * g represents the same rotation.
    
    :param q: (..., 4) - quaternion [w, x, y, z]
    :param symmetry: (N, 4) - symmetry group elements
    :return: (..., N, 4) - equivalent quaternions under symmetry
    """
    # q * g for all g in symmetry
    # q shape: (..., 4) → (..., 1, 4)
    # symmetry shape: (N, 4)
    # result: (..., N, 4)
    return quat_mul(q[..., tf.newaxis, :], symmetry)


def get_optimal_rotation_quat(v_ref: tf.Tensor, v_tar: tf.Tensor) -> tf.Tensor:
    """
    Compute the unit quaternion (w, x, y, z) that best rotates N reference
    vectors to N target vectors jointly in least-squares sense via Horn's method.

    Parameters
    ----------
    v_ref : tf.Tensor
        Reference vectors, shape `(..., N, 3)`. Leading `...` are batch-like axes;
        all N vectors share one quaternion. Must be broadcastable to `v_tar`.
    v_tar : tf.Tensor
        Target vectors, shape `(..., N, 3)`. The i-th vector in `v_ref` is matched
        to the i-th vector in `v_tar`.

    Returns
    -------
    tf.Tensor
        Unit quaternion(s) as (w, x, y, z), shape `(..., 4)`, where `...` is the
        common leading batch shape of the inputs.
    
    NOTE: This function uses Horn's K-matrix eigendecomposition method.
    Implementation using Kabsch's SVD method causes defect classification
    errors in get_defect_indices() due to different quaternion numerical precision.
    Do NOT replace with SVD-based approach as it breaks defect detection.
    """
    batch_shape = tf.broadcast_dynamic_shape(tf.shape(v_ref)[:-1], tf.shape(v_tar)[:-1])
    full_shape = tf.concat([batch_shape, [3]], axis=0)
    v_ref = tf.broadcast_to(v_ref, full_shape)
    v_tar = tf.broadcast_to(v_tar, full_shape)

    # --- Horn eigen formulation ------------------------------------------------
    H = tf.linalg.matmul(v_ref, v_tar, transpose_a=True)  # (...,3,3)

    # components for K  (see Horn 1987)
    h11, h12, h13 = H[..., 0, 0], H[..., 0, 1], H[..., 0, 2]
    h21, h22, h23 = H[..., 1, 0], H[..., 1, 1], H[..., 1, 2]
    h31, h32, h33 = H[..., 2, 0], H[..., 2, 1], H[..., 2, 2]

    K00 = h11 + h22 + h33
    K01 = h23 - h32
    K02 = h31 - h13
    K03 = h12 - h21

    K11 = h11 - h22 - h33
    K12 = h12 + h21
    K13 = h13 + h31

    K22 = -h11 + h22 - h33
    K23 = h23 + h32

    K33 = -h11 - h22 + h33

    K = tf.stack([
        tf.stack([K00, K01, K02, K03], -1),
        tf.stack([K01, K11, K12, K13], -1),
        tf.stack([K02, K12, K22, K23], -1),
        tf.stack([K03, K13, K23, K33], -1),
    ], -2)                                            # (...,4,4)

    # largest eigenvector → quaternion
    eig_vals, eig_vecs = tf.linalg.eigh(K)
    q = eig_vecs[..., -1]                              # (...,4)

    # normalise & ensure w ≥ 0 for a canonical sign
    q = tf.math.l2_normalize(q, axis=-1)
    q = tf.where(q[..., :1] < 0., -q, q)
    
    return q


def _find_nearest_quaternion(q1: tf.Tensor, q2: tf.Tensor = None, dtype=tf.int32) -> tuple:
    """
    Find the quaternion in q2 that is closest to q1.
    
    Finds the quaternion from q2 that has maximum dot product with q1,
    representing the most similar rotation.
    
    :param q1: (..., 4) - query quaternion. If None, uses identity [1,0,0,0]
    :param q2: (..., N, 4) - candidate quaternions
    :return: (nearest_quaternion (..., 4), nearest_index (...,))
    """
    # broadcast q1 and q2 to the same shape
    shape = tf.broadcast_dynamic_shape(tf.shape(q1)[:-1], tf.shape(q2)[:-2])
    q1 = tf.broadcast_to(q1, tf.concat([shape, [4]], axis=0))
    q2 = tf.broadcast_to(q2, tf.concat([shape, [tf.shape(q2)[-2]], [4]], axis=0))
    
    # Compute quaternion similarity as dot product
    # q1: (..., 4) → (..., 1, 4)
    # q2: (..., N, 4)
    # similarities: (..., N)
    similarities = tf.reduce_sum(q1[..., tf.newaxis, :] * q2, axis=-1)
    
    # Find index of maximum similarity
    nearest_index = tf.cast(tf.argmax(similarities, axis=-1), dtype=dtype)
    
    # Gather the nearest quaternion
    batch_dims = nearest_index.shape.rank
    if batch_dims is None:
        batch_dims = q2.shape.rank - 2
    if batch_dims is None:
        raise ValueError("Unable to infer static batch_dims for tf.gather.")
    nearest_quaternion = tf.gather(
        q2,
        nearest_index,
        axis=-2,
        batch_dims=int(batch_dims),
    )
    
    return nearest_quaternion, nearest_index


def _make_valid_site_mask_for_hexagonal_lattice(size1: int, size2: int = None, dtype=tf.float64):
    ii, jj = tf.meshgrid(
        tf.range(size1, dtype=dtype) - size1 // 2,
        tf.range(size2, dtype=dtype) - size2 // 2,
        indexing='ij'
    )
    mask = tf.cast(tf.where((ii - jj) % 3 == 0, 1., 0.), dtype=dtype)
    return mask


# @tf.function
def get_defect_hexagonal(spins: tf.Tensor, base_spin: tf.Tensor, group: tf.Tensor, conjugation_table: tf.Tensor, boundary_mask: tf.Tensor) -> tf.Tensor:
    """
    Classify defects in spin configuration.
    
    Determines defect type at each pixel by comparing spin configuration
    against symmetry group equivalents.
    
    :param spins: (..., Height, Width, n_spins, 3) - spin vectors
    :param conjugate: whether to return conjugacy class (True) or raw index (False)
    :return: (..., Height, Width) - defect type index
    
    CRITICAL: Uses Horn's K-matrix method for quaternion computation (via get_optimal_rotation_quat).
    Kabsch's SVD method causes incorrect defect classification due to quaternion precision differences.
    Do NOT use SVD approach as it breaks defect detection consistency.
    """
    base_spin = tf.broadcast_to(base_spin, tf.shape(spins))
    quaternion = get_optimal_rotation_quat(base_spin, spins)


    a_identities = _get_identical_quaternions(tf.roll(quaternion, 1, axis=-2), group)  # (..., Height, Width, N, 4)
    a, _ = _find_nearest_quaternion(
        tf.constant([1., 0., 0., 0.], dtype=spins.dtype),
        a_identities
    )

    b = tf.roll(tf.roll(a, 1, axis=-3), -1, axis=-2)
    c = tf.roll(b, -1, axis=-2)
    d = tf.roll(c, -1, axis=-3)
    e = tf.roll(tf.roll(d, -1, axis=-3), 1, axis=-2) 
    f = tf.roll(e, 1, axis=-2)

    loop = [b, c, d, e, f, a]

    for curr_q in loop:
        q_identities = _get_identical_quaternions(curr_q, group)
        nearest_q, idx = _find_nearest_quaternion(a, q_identities)
        a = nearest_q
    defect_type = tf.gather(conjugation_table, idx)
    mask = _make_valid_site_mask_for_hexagonal_lattice(spins.shape[-4], spins.shape[-3], dtype=tf.float64)
    defect_type = tf.where(mask > 0.5, defect_type, tf.zeros_like(defect_type))
    defect_type = tf.where(boundary_mask > 0.5, tf.zeros_like(defect_type), defect_type)
    return defect_type


# @tf.function
def get_defect(
    spins: tf.Tensor,
    base_spin: tf.Tensor,
    group: tf.Tensor,
    conjugation_table: tf.Tensor,
    dtype: tf.DType = tf.int32,
    boundary_mask: tf.Tensor = None,
) -> tf.Tensor:
    """
    Classify defects in spin configuration.
    
    Determines defect type at each pixel by comparing spin configuration
    against symmetry group equivalents.
    
    :param spins: (Height, Width, n_spins, 3) - spin vectors
    :param conjugate: whether to return conjugacy class (True) or raw index (False)
    :return: (Height, Width) - defect type index
    
    CRITICAL: Uses Horn's K-matrix method for quaternion computation (via get_optimal_rotation_quat).
    Kabsch's SVD method causes incorrect defect classification due to quaternion precision differences.
    Do NOT use SVD approach as it breaks defect detection consistency.
    """
    # Determine spin system type and symmetry group
    # Convert spins to quaternions using Horn's K-matrix method
    # (Kabsch's SVD method causes defect classification errors - DO NOT USE)
    # base_spins: (n_spins, 3), spins: (..., Height, Width, n_spins, 3)
    # get_optimal_rotation_quat accepts (..., N, 3) and returns (..., 4)
    quaternion = get_optimal_rotation_quat(base_spin, spins)  # (..., Height, Width, 4)

    a_identities = _get_identical_quaternions(quaternion, group)  # (..., Height, Width, N, 4)
    a, _ = _find_nearest_quaternion(
        tf.constant([1., 0., 0., 0.], dtype=spins.dtype),
        a_identities,
    )  # (..., Height, Width, 4)
    b = tf.roll(a, -1, axis=-2)
    c = tf.roll(b, -1, axis=-3)
    d = tf.roll(c, 1, axis=-2)

    a_identities = _get_identical_quaternions(a, group)
    b_identities = _get_identical_quaternions(b, group)
    c_identities = _get_identical_quaternions(c, group)
    d_identities = _get_identical_quaternions(d, group)

    # simplex 1
    identity = tf.constant([1.0, 0.0, 0.0, 0.0], dtype=spins.dtype)
    d_start, _ = _find_nearest_quaternion(identity, d_identities, dtype=dtype)
    a_nearest, _ = _find_nearest_quaternion(d_start, a_identities, dtype=dtype)
    b_nearest, _ = _find_nearest_quaternion(a_nearest, b_identities, dtype=dtype)
    d_nearest, element1 = _find_nearest_quaternion(b_nearest, d_identities, dtype=dtype)
    conjugacy_class1 = tf.gather(conjugation_table, element1)
    
    # simplex 2
    identity = tf.constant([1.0, 0.0, 0.0, 0.0], dtype=spins.dtype)
    d_start, _ = _find_nearest_quaternion(identity, d_identities, dtype=dtype)
    b_nearest, _ = _find_nearest_quaternion(d_start, b_identities, dtype=dtype)
    c_nearest, _ = _find_nearest_quaternion(b_nearest, c_identities, dtype=dtype)
    d_nearest, element2 = _find_nearest_quaternion(c_nearest, d_identities, dtype=dtype)
    conjugacy_class2 = tf.gather(conjugation_table, element2)

    conjugacy_class1 = tf.where(boundary_mask > 0.5, tf.zeros_like(conjugacy_class1), conjugacy_class1)
    conjugacy_class2 = tf.where(boundary_mask > 0.5, tf.zeros_like(conjugacy_class2), conjugacy_class2)

    conjugacy_class = tf.stack([conjugacy_class1, conjugacy_class2], axis=-1)
    return conjugacy_class

def defects_from_spin_directory(
    save_dir: str,
    N: int,
    iters_filename: str = "iters.npy",
    defects_filename: str = "defects.npy",
    boundary_mask: tf.Tensor | None = None,
    skip_ind: int = 1,
) -> dict[int, np.ndarray]:
    """
    Load ``save_dir/spin/{iter:07d}.npy`` and build defect classification fields only at timesteps
    where they may change, using **binary search on the time axis**.

    **Why binary search.** For each time interval ``[low, high]`` we compare defect fields at the
    endpoints (already known). If they are equal, nothing inside the interval is explored. If they
    differ, we split at ``mid = (low + high) // 2``, compute defects at ``mid``, and recurse on
    ``[low, mid]`` and/or ``[mid, high]`` only when the defect field still differs across a sub-interval.
    This avoids evaluating ``get_defect`` at every iteration ``0..N`` (which would be very expensive),
    but the **number of queue operations is not known in advance**; wall time can be large when many
    midpoints must be classified.

    Cached defect arrays are reused: if iteration ``i`` was already computed, it is not recomputed.

    Finally saves ``{save_dir}/{iters_filename}`` (T,) and ``{save_dir}/{defects_filename}`` (T, H, W, 2).

    :param save_dir: Root directory containing ``spin/`` (e.g. ``figures/figure2b``).
    :param N: Last timestep index (spins exist for indices ``0..N``).
    :param iters_filename: Output filename for the list of retained iteration indices.
    :param defects_filename: Output filename for stacked defect fields.
    :param boundary_mask: Optional mask passed to ``get_defect``.
    :param skip_ind: First iteration index used to seed the comparison at the left end (see call sites).
    :return: ``{iteration: defect_field}`` with boundary rows/columns zeroed in the saved arrays.
    """
    import os
    from collections import deque

    os.makedirs(save_dir, exist_ok=True)

    spin_sample = np.load(os.path.join(save_dir, "spin", "0000000.npy"))
    multi_spin_mode = "three_spin" if spin_sample.shape[-2] == 3 else "four_spin"
    base_spin = get_base_spin(multi_spin_mode=multi_spin_mode)
    group = get_group(multi_spin_mode=multi_spin_mode)
    conjugation_table = get_conjugation_table(multi_spin_mode=multi_spin_mode)

    cache: dict[int, np.ndarray] = {}
    # Counts only for progress logging (binary-search workload is data-dependent).
    stats = {"defect_evals": 0}  # number of full get_defect computations (cache misses)

    def _get_defect_at(i: int) -> np.ndarray:
        if i in cache:
            return cache[i]
        stats["defect_evals"] += 1
        spin = np.load(os.path.join(save_dir, "spin", f"{i:07d}.npy"))
        defect = np.array(get_defect(spin, base_spin, group, conjugation_table, boundary_mask=boundary_mask))
        cache[i] = defect

        return defect

    print(
        "[Binary search in progress] Refining defect snapshots over iterations. "
        "This may take a few hours.",
        flush=True,
    )

    defect_init = _get_defect_at(skip_ind)
    defect_final = _get_defect_at(N)

    # Work queue: each item is (low, high, defect_low, defect_high). Intervals with
    # identical endpoint defects are skipped without enqueueing children.
    que = deque([(0, N, defect_init, defect_final)])
    intervals_processed = 0

    while que:
        low, high, defect_low, defect_high = que.popleft()
        intervals_processed += 1
        if high <= low + 1:
            print(
                f"[Binary search in progress] remaining queue: {len(que)}  visited: {intervals_processed}",
                end="\r",
                flush=True,
            )
            continue
        mid = (low + high) // 2
        defect_mid = _get_defect_at(mid)

        if not np.array_equal(defect_low, defect_mid):
            que.append((low, mid, defect_low, defect_mid))
        if not np.array_equal(defect_mid, defect_high):
            que.append((mid, high, defect_mid, defect_high))

        print(
            f"remaining queue: {len(que)}  visited: {intervals_processed}",
            end="\r",
            flush=True,
        )

    print(
        f"\n[Binary search complete] intervals processed: {intervals_processed}, "
        f"defect evaluations (get_defect): {stats['defect_evals']}",
        flush=True,
    )

    out: dict[int, np.ndarray] = {}
    for k in sorted(cache.keys()):
        d = cache[k].copy()
        d[-1] = 0
        d[:, -1] = 0
        out[k] = d

    keys_sorted = sorted(out.keys())
    iters_arr = np.array(keys_sorted, dtype=np.int64)
    defects_arr = np.stack([out[k] for k in keys_sorted], axis=0)
    np.save(os.path.join(save_dir, iters_filename), iters_arr)
    np.save(os.path.join(save_dir, defects_filename), defects_arr)

    return out
