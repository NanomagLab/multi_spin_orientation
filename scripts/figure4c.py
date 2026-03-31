import os
import random
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from lib.spin_utils import get_chebychev_boundary_mask, defects_from_spin_directory, get_group
from lib.initial_conditions import get_two_defects_spin_fields, get_radial_spin
from lib.magnetism import update_spin
from lib.visualize import plot_trajectory, plot_spin
from lib.trajectory import get_trajectory

def main():
    RADIUS = 18
    DEFECT_POSITION = 6
    PLOT_RADIUS = 7
    ORIGIN_ON_GRID = True
    ITERATIONS = 20000
    IMAGE_SAVE_STEP = 1000
    MULTI_SPIN_MODE = "four_spin"
    NOISE_LEVEL = 0.01 # to break symmetry

    Z_END_LIST = [4000, 20000]
    Z_STEP_LIST = [2000, 10000]

    save_dir = "figures/figure4c"
    plot_mask = get_chebychev_boundary_mask(PLOT_RADIUS, ORIGIN_ON_GRID)
    truncate = RADIUS - PLOT_RADIUS

    # make initial two defects spin configurations
    names = ["a_ab_a", "ab"]

    for name in names:
        os.makedirs(f"{save_dir}/{name}/spin", exist_ok=True)
        os.makedirs(f"{save_dir}/{name}/image", exist_ok=True)

    group = get_group(multi_spin_mode=MULTI_SPIN_MODE)
    quats_list = [
        [group[1], group[15], group[6]], # a_ab_a
    ] # boundary, defect1, defect2

    print("Getting initial spins... this may take a few minutes.")
    init_spins, boundary_mask = get_two_defects_spin_fields(
        quats_list=quats_list,
        defect_position=DEFECT_POSITION,
        boundary_radius=RADIUS,
        multi_spin_mode=MULTI_SPIN_MODE,
    )
    
    ab_axis = np.array([1., 0., 0.])
    ab_angle = np.pi
    ab_spin = get_radial_spin(
        size=boundary_mask.shape[0],
        axis=ab_axis,
        angle=ab_angle,
        multi_spin_mode=MULTI_SPIN_MODE,
        origin_on_grid=ORIGIN_ON_GRID,
    )

    init_spins = np.stack([init_spins[0], ab_spin], axis=0)
    for name, spin in zip(names, init_spins):
        np.save(f"{save_dir}/{name}/spin/0000000.npy", spin)
        plot_spin(
            spin[truncate:-truncate, truncate:-truncate],
            multi_spin_mode=MULTI_SPIN_MODE,
            origin_on_grid=ORIGIN_ON_GRID,
            mask=plot_mask,
            view="top",
            radius=PLOT_RADIUS,
            save_path=f"{save_dir}/{name}/image/{0:07d}.png",
        )

    print("Relaxing spins... this may take a few minutes.")
    spins = init_spins
    # break symmetry
    spins = spins + NOISE_LEVEL * tf.random.normal(shape=spins.shape, dtype=spins.dtype)
    for i in tqdm(range(1, ITERATIONS + 1), desc="Relaxing spins"):
        spins = update_spin(
            spins,
            multi_spin_mode=MULTI_SPIN_MODE,
            fix_mask=boundary_mask,
        )
        for name, spin in zip(names, spins):
            np.save(f"{save_dir}/{name}/spin/{i:07d}.npy", spin)
            if i % IMAGE_SAVE_STEP != 0:
                continue
            plot_spin(
                spin[truncate:-truncate, truncate:-truncate],
                multi_spin_mode=MULTI_SPIN_MODE,
                origin_on_grid=ORIGIN_ON_GRID,
                mask=plot_mask,
                view="top",
                radius=PLOT_RADIUS,
                save_path=f"{save_dir}/{name}/image/{i:07d}.png",
            )

    print("Building defects/trajectories...")
    for name, z_end, z_step in zip(names, Z_END_LIST, Z_STEP_LIST):
        name_dir = f"{save_dir}/{name}"
        defect_dict = defects_from_spin_directory(name_dir, z_end, boundary_mask=boundary_mask)
        trajectory = get_trajectory(
            defect_dict,
            multi_spin_mode=MULTI_SPIN_MODE,
        )
        plot_trajectory(
            trajectory,
            multi_spin_mode=MULTI_SPIN_MODE,
            iterations=ITERATIONS,
            z_limit=[0, z_end],
            z_step=z_step,
            radius=RADIUS,
            save_path=f"{name_dir}/defect_trajectories.png",
        )
    

    
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)
    tf.config.experimental.enable_op_determinism()
    main()