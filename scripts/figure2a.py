import numpy as np
import os
from lib.initial_conditions import get_radial_spin
from lib.spin_utils import get_chebychev_boundary_mask
from lib.visualize import plot_spin

def main():
    RADIUS = 6
    ORIGIN_ON_GRID = False
    MULTI_SPIN_MODE = "three_spin"

    save_dir = "figures/figure2a"
    os.makedirs(save_dir, exist_ok=True)

    boundary_mask = get_chebychev_boundary_mask(RADIUS, ORIGIN_ON_GRID)

    # make initial raidal spin configurations
    configuration_names = ["e", "r", "s", "s_inv"]
    
    r_axis = np.array([0., 0., 1.])
    r_angle = 2. / 3. * np.pi
    e_spin = get_radial_spin(
        size=boundary_mask.shape[0],
        axis=np.array([0., 0., 1.]),
        angle=0.,
        multi_spin_mode=MULTI_SPIN_MODE,
        origin_on_grid=ORIGIN_ON_GRID,
    )
    r_spin = get_radial_spin(
        size=boundary_mask.shape[0],
        axis=r_axis,
        angle=r_angle,
        multi_spin_mode=MULTI_SPIN_MODE,
        origin_on_grid=ORIGIN_ON_GRID,
    )
    s_axis = np.array([1., 0., 0.])
    s_angle = np.pi
    s_spin = get_radial_spin(
        size=boundary_mask.shape[0],
        axis=s_axis,
        angle=s_angle,
        multi_spin_mode=MULTI_SPIN_MODE,
        origin_on_grid=ORIGIN_ON_GRID,
    )
    s_inv_spin = get_radial_spin(
        size=boundary_mask.shape[0],
        axis=s_axis,
        angle=-s_angle,
        multi_spin_mode=MULTI_SPIN_MODE,
        origin_on_grid=ORIGIN_ON_GRID,
    )
    init_spins = np.stack([e_spin, r_spin, s_spin, s_inv_spin], axis=0)
    for name, spin in zip(configuration_names, init_spins):
        plot_spin(
            spin,
            multi_spin_mode=MULTI_SPIN_MODE,
            origin_on_grid=ORIGIN_ON_GRID,
            mask=boundary_mask,
            view="side",
            save_path=f'{save_dir}/{name}.png',
        )


    
if __name__ == "__main__":
    main()