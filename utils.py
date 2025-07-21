"""
Utility helpers and visualisation tools for the flat-quadtree Barnes–Hut solver.
"""

from __future__ import annotations

import os
os.environ["PATH"] = os.path.join(os.getcwd(), "ffmpeg", "ffmpeg-2025-07-17-git-bc8d06d541-full_build", "bin") + os.pathsep + os.environ["PATH"]
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from tqdm import tqdm

from array_quadtree import FlatQuadTree
from body import Body

# --------------------------------------------------------------------------- #
# Initial Conditions                                                          #
# --------------------------------------------------------------------------- #

# --- galaxy generator with spin control --------------------------------
def generate_galaxy(
    r0: float,
    total_mass: float,
    n_bodies: int,
    half_box: float,
    *,
    center: tuple[float, float] = (0.0, 0.0),
    vel_boost: float = 8.0,
    spin: int = +1,
) -> list[Body]:
    """Exponential disk orbiting **clockwise** (*spin=+1*) or
    **counter-clockwise** (*spin=-1*) about *center*.
    """
    cx, cy = center
    mass   = total_mass / n_bodies
    bodies: list[Body] = []

    for _ in range(n_bodies):
        r = -r0 * np.log(1.0 - np.random.rand())
        if r >= half_box:
            continue

        phi = 2 * np.pi * np.random.rand()
        rx, ry = r * np.cos(phi), r * np.sin(phi)

        v = vel_boost * np.exp(-r0 / r) / np.sqrt(r)
        vx = -spin * v * np.sin(phi)
        vy =  spin * v * np.cos(phi)

        bodies.append(Body(mass,
                           cx + rx, cy + ry,
                           vx, vy,
                           box_half=half_box))
    return bodies


# --- optional random-grid generator ------------------------------------
def generate_random_grid(
    m_tot: float,
    n_bodies: int,
    half_box: float,
    v_factor: float = 0.01,
) -> list[Body]:
    bodies: list[Body] = []
    mass = m_tot / n_bodies
    side = int(np.ceil(np.sqrt(n_bodies)))
    spacing = (2 * half_box) / side
    jitter = 0.4 * spacing

    count = 0
    for iy in range(side):
        for ix in range(side):
            if count >= n_bodies:
                break
            rx = -half_box + (ix + 0.5) * spacing
            ry = -half_box + (iy + 0.5) * spacing
            rx += (np.random.rand() - 0.5) * 2 * jitter
            ry += (np.random.rand() - 0.5) * 2 * jitter
            vx, vy = np.random.randn(2) * v_factor
            bodies.append(Body(mass, rx, ry, vx, vy, box_half=half_box))
            count += 1
        if count >= n_bodies:
            break
    return bodies

# --------------------------------------------------------------------------- #
# One Barnes–Hut Euler–Cromer step (array backend)                            #
# --------------------------------------------------------------------------- #
def bh_step(
    bodies: list[Body],
    half_box: float,
    theta: float,
    epsilon: float,
    dt: float,
    *,
    tree_buffer: Optional[FlatQuadTree] = None,
) -> Tuple[list[Body], FlatQuadTree]:
    """Advance *bodies* by one step using the array-based Barnes–Hut solver.

    Args:
        bodies:      Particle list (updated in place).
        half_box:    Half-length *L* of the periodic square domain.
        theta:       Barnes–Hut opening angle.
        epsilon:     Softening length.
        dt:          Time increment.
        tree_buffer: Optional reusable :class:`FlatQuadTree`.

    Returns:
        Tuple ``(bodies, tree_buffer)`` – updated list and reusable tree.
    """

    pos, vel, mass = Body.to_arrays(bodies)

    if tree_buffer is None:
        tree_buffer = FlatQuadTree(half_box=half_box, max_bodies=len(bodies))
    tree_buffer.build(pos, mass)
    forces = tree_buffer.forces(pos, mass, theta=theta, eps=epsilon)

    vel += (forces / mass[:, None]) * dt
    pos = np.mod(pos + vel * dt + half_box, 2.0 * half_box) - half_box
    Body.from_arrays(bodies, pos, vel)
    return bodies, tree_buffer


# --------------------------------------------------------------------------- #
# Visualization with optional MP4 video export (scatter or density plot)      #
# --------------------------------------------------------------------------- #
def run_sim(
    steps: int,
    bodies: list[Body],
    half_box: float,
    theta: float,
    epsilon: float,
    dt: float,
    *,
    point_size: int = 3,
    plot_type: str = "scatter",
    outfile: str = "output/galaxy.mp4",
    fps: int = 30,
) -> None:
    """Render the simulation to video and save it as an MP4 file.

    The function mirrors the real-time visualiser but captures each frame
    directly from the PyQtGraph window and pipes it into Matplotlib’s
    ``FFMpegWriter``. Both “scatter’’ and log-scaled “density’’ modes are
    supported, with the latter producing a normalised heat-map.

    Args:
        steps:       Number of integration steps to record.
        bodies:      Particle list (mutated in place).
        half_box:    Half-length *L* of the simulation square.
        theta:       Barnes–Hut opening angle.
        epsilon:     Softening length.
        dt:          Time increment.
        point_size:  Marker size for the scatter plot.
        plot_type:   Either ``"scatter"`` or ``"density"``.
        outfile:     Output file path (written inside *output/* directory).
        fps:         Frames per second for the resulting video.
    """
    outfile = os.path.join("output", os.path.basename(outfile))
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(title="Barnes–Hut Video Export")
    win.resize(1920, 1080)
    view = win.addViewBox(lockAspect=True)
    view.setRange(QtCore.QRectF(-half_box, -half_box, 2 * half_box, 2 * half_box))

    if plot_type == "scatter":
        plot_item = pg.ScatterPlotItem(size=point_size, pen=None, brush=pg.mkBrush("w"))
        res = None
    else:
        plot_item = pg.ImageItem()
        cmap = pg.colormap.get("inferno")
        plot_item.setLookupTable(cmap.getLookupTable())
        res = 512
        pixel = (2 * half_box) / res
        plot_item.setPos(-half_box, -half_box)
        plot_item.setScale(pixel)
    view.addItem(plot_item)
    win.show()

    img_format = (
        QtGui.QImage.Format_ARGB32
        if hasattr(QtGui.QImage, "Format_ARGB32")
        else QtGui.QImage.Format.Format_ARGB32
    )

    def grab_rgb() -> np.ndarray:
        pix = win.grab()
        qimg = pix.toImage().convertToFormat(img_format)
        h, w, stride = qimg.height(), qimg.width(), qimg.bytesPerLine()
        buf = qimg.bits()
        size = (
            qimg.sizeInBytes()  # Qt6
            if hasattr(qimg, "sizeInBytes")
            else stride * h     # Qt5 fallback
        )
        buf.setsize(size)
        arr = np.frombuffer(buf, np.uint8).reshape(h, stride // 4, 4)
        return arr[:, :w, :3].copy()

    frame_rgb = grab_rgb()
    h_px, w_px = frame_rgb.shape[:2]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(w_px / fig.dpi, h_px / fig.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(frame_rgb)
    writer = FFMpegWriter(fps=fps)
    writer.setup(fig, outfile, dpi=fig.dpi)

    tree_buf: Optional[FlatQuadTree] = None

    for _ in tqdm(range(steps), desc="Writing frames"):
        bodies, tree_buf = bh_step(
            bodies,
            half_box=half_box,
            theta=theta,
            epsilon=epsilon,
            dt=dt,
            tree_buffer=tree_buf,
        )
        pos = np.asarray([b.r for b in bodies], dtype=np.float32)

        if plot_type == "scatter":
            plot_item.setData(pos[:, 0], pos[:, 1])
        else:
            H, _, _ = np.histogram2d(
                pos[:, 0], pos[:, 1],
                bins=res,
                range=[[-half_box, half_box], [-half_box, half_box]],
            )
            H_log = np.log1p(H)
            H_norm = H_log / (H_log.max() if H_log.max() > 0 else 1.0)
            plot_item.setImage(H_norm, levels=(0.0, 1.0), autoLevels=False)

        QtWidgets.QApplication.processEvents()
        frame_rgb = grab_rgb()
        im.set_data(frame_rgb)
        writer.grab_frame()

    writer.finish()
    plt.close(fig)
    (app.exec if hasattr(app, "exec") else app.exec_)()
