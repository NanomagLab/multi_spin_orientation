import os
import random
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import tensorflow as tf
from lib.initial_conditions import get_radial_spin
from lib.spin_utils import get_chebychev_boundary_mask
from lib.visualize import plot_spin, plot_trajectory
from lib.magnetism import update_spin
from lib.trajectory import get_trajectory
from lib.spin_utils import defects_from_spin_directory
from tqdm import tqdm

def main():
    RADIUS = 32
    PLOT_RADIUS = 7
    ORIGIN_ON_GRID = True
    ITERATIONS = 20000
    IMAGE_SAVE_STEP = 1000
    MULTI_SPIN_MODE = "four_spin"
    NOISE_LEVEL = 0.01 # to break symmetry

    Z_END_LIST = [20000, 20000, 20000]
    Z_STEP_LIST = [10000, 10000, 10000]

    save_dir = "figures/figure5b"
    boundary_mask = get_chebychev_boundary_mask(RADIUS, ORIGIN_ON_GRID)
    plot_mask = get_chebychev_boundary_mask(PLOT_RADIUS, ORIGIN_ON_GRID)
    truncate = RADIUS - PLOT_RADIUS

    # make initial raidal spin configurations
    configuration_names = ["a2", "a3", "abab"]

    for name in configuration_names:
        os.makedirs(f"{save_dir}/{name}/spin", exist_ok=True)
        os.makedirs(f"{save_dir}/{name}/image", exist_ok=True)

    a_axis = np.array([1., 1., 1.]) / np.sqrt(3.)
    a_angle = 2. / 3. * np.pi
    a2_spin = get_radial_spin(
        size=boundary_mask.shape[0],
        axis=np.array(a_axis),
        angle=a_angle * 2.,
        multi_spin_mode=MULTI_SPIN_MODE,
        origin_on_grid=ORIGIN_ON_GRID,
    )
    a3_spin = get_radial_spin(
        size=boundary_mask.shape[0],
        axis=np.array(a_axis),
        angle=a_angle * 3.,
        multi_spin_mode=MULTI_SPIN_MODE,
        origin_on_grid=ORIGIN_ON_GRID,
    )
    ab_axis = np.array([1., 0., 0.])
    ab_angle = np.pi
    abab_spin = get_radial_spin(
        size=boundary_mask.shape[0],
        axis=ab_axis,
        angle=ab_angle * 2.,
        multi_spin_mode=MULTI_SPIN_MODE,
        origin_on_grid=ORIGIN_ON_GRID,
    )
    init_spins = np.stack([a2_spin, a3_spin, abab_spin], axis=0)
    for name, spin in zip(configuration_names, init_spins):
        np.save(f"{save_dir}/{name}/spin/0000000.npy", spin)
        plot_spin(
            spin[truncate:-truncate, truncate:-truncate],
            multi_spin_mode=MULTI_SPIN_MODE,
            origin_on_grid=ORIGIN_ON_GRID,
            mask=plot_mask,
            view="top",
            radius=PLOT_RADIUS,
            save_path=f'{save_dir}/{name}/image/{0:07d}.png',
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
        for name, spin in zip(configuration_names, spins):
            np.save(f"{save_dir}/{name}/spin/{i:07d}.npy", spin)
            if i % IMAGE_SAVE_STEP == 0:
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
    for name, z_end, z_step in zip(configuration_names, Z_END_LIST, Z_STEP_LIST):
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