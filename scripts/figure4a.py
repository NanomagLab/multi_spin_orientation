import numpy as np
import os
from lib.initial_conditions import get_radial_spin
from lib.spin_utils import get_chebychev_boundary_mask
from lib.visualize import plot_spin

def main():
    RADIUS = 6
    ORIGIN_ON_GRID = False
    MULTI_SPIN_MODE = "four_spin"

    save_dir = "figures/figure4a"
    os.makedirs(save_dir, exist_ok=True)

    boundary_mask = get_chebychev_boundary_mask(RADIUS, ORIGIN_ON_GRID)

    # make initial raidal spin configurations
    configuration_names = ["e", "a", "b", "ab"]

    a_axis = np.array([1., 1., 1.]) / np.sqrt(3.)
    a_angle = 2. / 3. * np.pi
    e_spin = get_radial_spin(
        size=boundary_mask.shape[0],
        axis=np.array([0., 0., 1.]),
        angle=0.,
        multi_spin_mode=MULTI_SPIN_MODE,
        origin_on_grid=ORIGIN_ON_GRID,
    )
    a_spin = get_radial_spin(
        size=boundary_mask.shape[0],
        axis=a_axis,
        angle=a_angle,
        multi_spin_mode=MULTI_SPIN_MODE,
        origin_on_grid=ORIGIN_ON_GRID,
    )
    b_axis = np.array([1., -1., 1.]) / np.sqrt(3.)
    b_angle = 2. / 3. * np.pi
    b_spin = get_radial_spin(
        size=boundary_mask.shape[0],
        axis=b_axis,
        angle=b_angle,
        multi_spin_mode=MULTI_SPIN_MODE,
        origin_on_grid=ORIGIN_ON_GRID,
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
    init_spins = np.stack([e_spin, a_spin, b_spin, ab_spin], axis=0)
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