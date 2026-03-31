import os
import random
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from lib.spin_utils import get_chebychev_boundary_mask, defects_from_spin_directory
from lib.initial_conditions import get_random_spin_fields
from lib.magnetism import update_spin
from lib.visualize import plot_trajectory, plot_spin
from lib.trajectory import get_trajectory

def main():
    RADIUS = 32
    ORIGIN_ON_GRID = True
    ITERATIONS = 30000
    MULTI_SPIN_MODE = "four_spin"
    IMAGE_SAVE_STEP = 1000

    Z_END = 30000
    Z_STEP = 15000

    save_dir = "figures/figure5c"

    os.makedirs(f"{save_dir}/spin", exist_ok=True)
    os.makedirs(f"{save_dir}/image", exist_ok=True)

    # make initial random spin configurations
    init_spin = get_random_spin_fields(
        radius=RADIUS,
        multi_spin_mode=MULTI_SPIN_MODE,
    )
    np.save(f"{save_dir}/spin/0000000.npy", init_spin)
    boundary_mask = get_chebychev_boundary_mask(RADIUS, ORIGIN_ON_GRID)
    plot_spin(
        init_spin,
        multi_spin_mode=MULTI_SPIN_MODE,
        origin_on_grid=ORIGIN_ON_GRID,
        mask=boundary_mask,
        view="top",
        radius=RADIUS,
        save_path=f"{save_dir}/image/0000000.png",
    )

    print("Relaxing spins... this may take a few minutes.")
    spins = init_spin
    for i in tqdm(range(1, ITERATIONS + 1), desc="Relaxing spins"):
        spins = update_spin(
            spins,
            multi_spin_mode=MULTI_SPIN_MODE,
            fix_mask=boundary_mask,
        )
        np.save(f"{save_dir}/spin/{i:07d}.npy", spins)
        if i % IMAGE_SAVE_STEP == 0:
            plot_spin(
                spins,
                multi_spin_mode=MULTI_SPIN_MODE,
                origin_on_grid=ORIGIN_ON_GRID,
                mask=boundary_mask,
                view="top",
                radius=RADIUS,
                save_path=f"{save_dir}/image/{i:07d}.png",
            )
    
    print("Computing defects and trajectories... This may take a few hours.")
    defect_dict = defects_from_spin_directory(save_dir, Z_END, boundary_mask=boundary_mask)
    # defects = np.load(f"{save_dir}/defects.npy")
    # iters = np.load(f"{save_dir}/iters.npy")
    # defect_dict = {iter: defect for iter, defect in zip(iters, defects)}
    
    trajectory = get_trajectory(
        defect_dict,
        multi_spin_mode=MULTI_SPIN_MODE,
    )
    plot_trajectory(
        trajectory,
        iterations=ITERATIONS,
        radius=RADIUS,
        multi_spin_mode=MULTI_SPIN_MODE,
        save_path=f"{save_dir}/defect_trajectories.png",     
        fig_width=900,
        fig_height=900,
        z_limit=[0, Z_END],
        z_step=Z_STEP,
    )
    
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)
    tf.config.experimental.enable_op_determinism()
    main()