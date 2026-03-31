from __future__ import annotations

from pickle import NONE
from mayavi import mlab
import numpy as np
import tensorflow as tf

from lib.spin_utils import get_positions
from lib.trajectory import Trajectory
from lib.magnetism import get_energy_density_multi_spin



def plot_three_spin(
        spin: np.ndarray|tf.Tensor,
        origin_on_grid: bool = True,
        mask: np.ndarray|tf.Tensor|None = None,
        save_path: str|None = None,
        view: str = "side",
        radius: float = None,
):
    """
    Visualize three-spin system configuration using Mayavi.
    
    :param spin: (Height, Width, n_spins, 3) - spin configuration (no batch dimension)
    :param origin_on_grid: whether to place the origin on the grid
    :param mask: optional mask to zero out certain spins
    :param save_path: path to save visualization, if None shows interactive plot
    :param view: viewing angle, one of "top", "side"
    :param lattice_mode: lattice mode
    """

    # Apply mask if provided
    if mask is not None:
        spin_masked = spin * (1. - mask[..., tf.newaxis, tf.newaxis])
    else:
        spin_masked = spin
    
    

    # Set background color white
    fig_size = (800, 800)
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=fig_size)
    
    # Create coordinate meshgrid
    yy, xx = get_positions(spin.shape[0], origin_on_grid=origin_on_grid, dtype=spin.dtype)
    
    # Convert spin configuration to xyz coordinates
    spin_masked = np.array(spin_masked)
    mesh_size = 0.4
    x = spin_masked[..., 0] * mesh_size + xx[..., np.newaxis]
    y = spin_masked[..., 1] * mesh_size + yy[..., np.newaxis]
    z = spin_masked[..., 2] * mesh_size
    
    # Flatten coordinates
    x = np.reshape(x, (-1,))
    y = np.reshape(y, (-1,))
    z = np.reshape(z, (-1,))
    
    # Calculate energy density
    energy_density = get_energy_density_multi_spin(spin, multi_spin_mode="three_spin")
    
    # Set default energy bounds
    energy_high = -38.46076882753714
    energy_low = -40.5
    
    # Normalize energy for coloring
    scalar = tf.clip_by_value(energy_density, energy_low, energy_high)
    scalar = np.stack([scalar, scalar, scalar], axis=-1)
    scalar = np.reshape(scalar, (-1,))
    
    # Create triangle mesh
    triangles = np.reshape(np.arange(len(x)), (-1, 3))
    mesh = mlab.triangular_mesh(
        x, y, z, triangles,
        scalars=scalar, vmin=energy_low, vmax=energy_high
    )
    
    # Add wireframe overlay
    mesh_frame = mlab.triangular_mesh(
        x, y, z, triangles,
        representation='wireframe',
        color=(0, 0, 0),
        line_width=2.0,
    )
    if view == "top":
        azimuth=0
        elevation=0
        
    elif view == "side":
        azimuth=-80
        elevation=60
        
        # show axes
        mlab.orientation_axes()
    else:
        raise ValueError(f"Invalid view: {view}. Must be one of 'top', 'side'.")
    
    # Save or display
    if save_path is None:
        mlab.show()
    else:
        if radius is not None:
            distance = radius * 4.
        else:
            distance = None
        mlab.view(azimuth=azimuth, elevation=elevation, distance=distance)
        mlab.savefig(save_path, size=(fig_size[0] * 2, fig_size[1] * 2))
        mlab.close(all=True)


def plot_four_spin(
    spin: np.ndarray|tf.Tensor,
    origin_on_grid: bool = True,
    mask: np.ndarray|tf.Tensor|None = None,
    save_path: str|None = None,
    view: str = "side",
    radius: float = None,
):
    """
    Visualize four-spin system configuration using Mayavi.
    
    :param spin: (Height, Width, n_spins, 3) - spin configuration (no batch dimension)
    :param origin_on_grid: whether to place the origin on the grid
    :param mask: optional mask to zero out certain spins
    :param save_path: path to save visualization, if None shows interactive plot
    :param view: viewing angle, one of "top", "side"
    :param lattice_mode: lattice mode
    """
    # Apply mask if provided
    if mask is not None:
        spin_masked = spin * (1. - mask[..., tf.newaxis, tf.newaxis])
    else:
        spin_masked = spin
    


    # Set background color white
    fig_size = (800, 800)
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=fig_size)

    mesh_size = 0.4
    yy, xx = get_positions(spin.shape[0], origin_on_grid=origin_on_grid, dtype=spin.dtype)

    spin_np = np.array(spin_masked)
    # Vertex positions: for each (i,j) and each of 4 spins, (x,y,z) = grid + spin*scale
    x = spin_np[..., 0] * mesh_size + xx[..., np.newaxis]
    y = spin_np[..., 1] * mesh_size + yy[..., np.newaxis]
    z = spin_np[..., 2] * mesh_size

    height, width = spin.shape[0], spin.shape[1]
    n_cells = height * width
    x_flat = np.reshape(x, (-1,))
    y_flat = np.reshape(y, (-1,))
    z_flat = np.reshape(z, (-1,))

    # Energy density per site (Height, Width)
    energy_density = get_energy_density_multi_spin(spin, multi_spin_mode="four_spin")
    energy_density = np.array(energy_density)
    energy_high = 23.318448407190903
    energy_low = 16.000000000000007
    energy_high = (energy_low + energy_high) / 2.

    scalar_site = np.clip(energy_density, energy_low, energy_high)
    # One scalar per vertex: repeat for 4 vertices per site -> (H, W, 4) -> flatten
    scalar_vertex = np.stack([scalar_site, scalar_site, scalar_site, scalar_site], axis=-1)
    scalar_flat = np.reshape(scalar_vertex, (-1,))

    # Tetrahedron: 4 faces, each a triangle (3 vertices). Vertices per cell: 0,1,2,3.
    # Faces: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    triangles = []
    for n in range(n_cells):
        base = n * 4
        triangles.append([base, base + 1, base + 2])
        triangles.append([base, base + 1, base + 3])
        triangles.append([base, base + 2, base + 3])
        triangles.append([base + 1, base + 2, base + 3])
    triangles = np.array(triangles)

    mesh = mlab.triangular_mesh(
        x_flat, y_flat, z_flat, triangles,
        scalars=scalar_flat, vmin=energy_low, vmax=energy_high
    )
    mesh_frame = mlab.triangular_mesh(
        x_flat, y_flat, z_flat, triangles,
        representation='wireframe',
        color=(0, 0, 0),
        line_width=1.0,
    )

    if view == "top":
        azimuth=0
        elevation=0

    elif view == "side":
        azimuth=-80
        elevation=60

        # show axes
        mlab.orientation_axes()
    else:
        raise ValueError(f"Invalid view: {view}. Must be one of 'top', 'side'.")

    if save_path is None:
        mlab.show()
    else:
        if radius is not None:
            distance = radius * 4.
        else:
            distance = None
        mlab.view(azimuth=azimuth, elevation=elevation, distance=distance)
        mlab.savefig(save_path)
        mlab.close(all=True)


def plot_spin(
    spin: np.ndarray|tf.Tensor,
    multi_spin_mode: str,
    origin_on_grid: bool,
    mask: np.ndarray|tf.Tensor,
    save_path: str|None = None,
    view: str = "top",
    radius: float = None,
):
    if multi_spin_mode == "three_spin":
        plot_three_spin(spin, origin_on_grid=origin_on_grid, mask=mask, save_path=save_path, view=view, radius=radius)
    elif multi_spin_mode == "four_spin":
        plot_four_spin(spin, origin_on_grid=origin_on_grid, mask=mask, save_path=save_path, view=view, radius=radius)
    else:
        raise ValueError(f"Invalid multi_spin_mode: {multi_spin_mode}. Must be one of 'three_spin', 'four_spin'.")



def plot_trajectory(
    trajectories: list[Trajectory],
    iterations: int,
    radius: float,
    multi_spin_mode: str,
    save_path: str | None = None,
    fig_width: int = 300,
    fig_height: int = 300,
    font_size: int = 18,
    z_limit: tuple[float, float] | None = None,
    z_step: float | None = None,
):
    """
    :param z_limit: z(t) axis range ``[z_start, z_end]``. If None, use ``[0, iterations]``.
    :param z_step: Tick spacing on the z axis. If None, use ``(z_end - z_start) / 5`` with a minimum of 1.
    """
    import plotly.graph_objects as go

    if multi_spin_mode == "three_spin":
        type_names = ["r", "r2", "r3", "s", "s3"]
        type_colors = ["red", "yellow", "purple", "green", "blue"]


    elif multi_spin_mode == "four_spin":
        type_names = ["a", "a2", "a3", "b2", "b", "ab"]
        type_colors = ["red", "yellow", "purple", "orange", "green", "blue"]
    else:
        raise ValueError("Invalid mode. Only 'three_spin' and 'four_spin' are supported.")

    trajectories_by_type = [[] for _ in range(len(type_names))]
    for traj in trajectories:
        trajectories_by_type[traj.type - 1].append(traj.traj)

    plot_limit = int(radius * 1.1 + 1)
    xlim = [-plot_limit, plot_limit]
    ylim = [-plot_limit, plot_limit]

    if z_limit is None:
        z_start, z_end = 0.0, float(iterations)
    else:
        z_start, z_end = float(z_limit[0]), float(z_limit[1])
        if z_start > z_end:
            z_start, z_end = z_end, z_start
    zlim = [z_start, z_end]

    span = max(z_end - z_start, 1e-12)
    if z_step is None:
        step = max(span / 5.0, 1.0)
    else:
        step = max(float(z_step), 1e-12)
    z_ticks = np.arange(z_start, z_end + step * 0.5, step)
    z_ticks = z_ticks[z_ticks <= z_end + 1e-9]
    if z_ticks.size == 0:
        z_ticks = np.array([z_start, z_end])
    elif abs(z_ticks[-1] - z_end) > 1e-6 * max(abs(z_end), 1.0):
        z_ticks = np.unique(np.append(z_ticks, z_end))

    fig = go.Figure()

    cx = 0.0
    cy = 0.0
    z_plane = float(z_end)
    angles = np.pi / 2.0 + np.linspace(0, 2 * np.pi, 7)[:-1]
    vx = cx + float(radius) * np.cos(angles)
    vy = cy + float(radius) * np.sin(angles)

    def inside_hexagon(px, py):
        for i in range(6):
            ax, ay = vx[i], vy[i]
            bx, by = vx[(i + 1) % 6], vy[(i + 1) % 6]
            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            if cross < 0:
                return False
        return True

    n = 120
    x_lin = np.linspace(cx - radius, cx + radius, n)
    y_lin = np.linspace(cy - radius, cy + radius, n)
    X, Y = np.meshgrid(x_lin, y_lin)
    inside = np.vectorize(inside_hexagon)(X, Y)
    Z = np.where(inside, z_plane, np.nan)
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            showscale=False,
            opacity=0.2,
            surfacecolor=np.where(inside, 0.0, np.nan),
            colorscale=[[0, "rgba(120,120,120,1)"], [1, "rgba(120,120,120,1)"]],
            name="mask",
            hoverinfo="skip",
        )
    )
    

    for typ, trajs in enumerate(trajectories_by_type):
        if typ >= len(type_colors):
            break
        xs, ys, zs = [], [], []
        for traj in trajs:
            if not traj:
                continue
            for (t, y, x) in traj:
                if not (zlim[0] <= t <= zlim[1]):
                    continue
                xs.append(float(x))
                ys.append(float(y))
                zs.append(float(t))
            xs.append(None)
            ys.append(None)
            zs.append(None)
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name=type_names[typ],
                line=dict(color=type_colors[typ], width=2),
            )
        )

    scene = dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="t",
        aspectmode="manual",
    )
    ticklen = 14

    x0, x1 = float(xlim[0]), float(xlim[1])
    xc = 0.5 * (x0 + x1)
    dx = x1 - x0
    scene["xaxis"] = dict(
        range=[x0, x1],
        tickmode="array",
        tickvals=[x0, xc, x1],
        ticktext=[f"{-dx/2:g}", "0", f"{dx/2:g}"],
        ticklen=ticklen,
    )

    y0, y1 = float(ylim[0]), float(ylim[1])
    yc = 0.5 * (y0 + y1)
    dy = y1 - y0
    scene["yaxis"] = dict(
        range=[y0, y1],
        tickmode="array",
        tickvals=[yc, y1],
        ticktext=["0", f"{dy / 2:g}"],
        ticklen=ticklen,
    )

    z0p = float(zlim[0])
    z1p = float(zlim[1])
    scene["zaxis"] = dict(
        range=[z1p, z0p],
        autorange=False,
        ticklen=ticklen,
        tickmode="array",
        tickvals=[float(v) for v in z_ticks],
        ticktext=[
            (f"{v / 1000.0:g}k" if abs(v) >= 1000 else f"{v:g}") for v in z_ticks
        ],
    )
    scene["aspectratio"] = dict(x=1.0, y=1.0, z=1.5)

    cam_eye = dict(x=1.6, y=1.6, z=1.6)
    dz = abs(zlim[1] - zlim[0])
    m = max(dx, dy, dz, 1.0)
    cam_eye = dict(
        x=max(2.0, 2.4 * dx / m),
        y=max(2.0, 2.4 * dy / m),
        z=max(1.2, 1.6 * dz / m),
    )

    layout_kw = dict(
        scene=scene,
        scene_camera=dict(eye=cam_eye),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(size=font_size),
        showlegend=False,
    )
    if fig_width is not None:
        layout_kw["width"] = fig_width
    if fig_height is not None:
        layout_kw["height"] = fig_height
    fig.update_layout(**layout_kw)

    if save_path is not None:
        sp = str(save_path)
        if sp.lower().endswith(".html"):
            fig.write_html(sp)
        else:
            kw = {"scale": 2}
            if fig_width is not None:
                kw["width"] = int(fig_width)
            if fig_height is not None:
                kw["height"] = int(fig_height)
            fig.write_image(sp, **kw)

    # fig.show()
    return fig

