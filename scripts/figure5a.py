import numpy as np
import os
from lib.initial_conditions import get_radial_spin
from lib.spin_utils import get_chebychev_boundary_mask
from lib.visualize import plot_spin

def main():
    RADIUS = 6
    ORIGIN_ON_GRID = False
    MULTI_SPIN_MODE = "four_spin"

    save_dir = "figures/figure5a"
    os.makedirs(save_dir, exist_ok=True)

    boundary_mask = get_chebychev_boundary_mask(RADIUS, ORIGIN_ON_GRID)

    # make initial raidal spin configurations
    configuration_names = ["a2", "a3", "abab"]

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