"""
Real-time driver for the array-based Barnes–Hut simulation with PyQtGraph.
"""

from __future__ import annotations

import numpy as np
from body import Body
from utils import run_sim


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

# ------------------------------------------------------------------------
if __name__ == "__main__":
    # global simulation parameters
    half_box = np.float32(100.0)
    theta    = np.float32(0.7)
    epsilon  = np.float32(0.05)
    dt       = np.float32(0.05)
    steps    = 5000
    point_sz = 1

    # galaxy settings
    n_each   = 10_000
    r0       = np.float32(12.5)
    m_disk   = np.float32(50.0)
    offset_x = np.float32(40.0)
    offset_y = np.float32(10.0)
    v_bulk   = np.float32(0.6)
    v_boost  = 10.0
    spin     = +1

    # generate a single galaxy at the origin
    base = generate_galaxy(r0, m_disk, n_each, half_box,
                           center=(0.0, 0.0),
                           vel_boost=v_boost,
                           spin=spin)

    # create galaxy A by copying base and shifting by (−x, −y)
    gal_A: list[Body] = []
    for b in base:
        gal_A.append(Body(b.m,
                          b.r[0] - offset_x, b.r[1] - offset_y,
                          b.v[0] + v_bulk, b.v[1],
                          box_half=half_box))

    # create galaxy B by copying base and shifting by (+x, +y)
    gal_B: list[Body] = []
    for b in base:
        gal_B.append(Body(b.m,
                          b.r[0] + offset_x, b.r[1] + offset_y,
                          b.v[0] - v_bulk, b.v[1],
                          box_half=half_box))

    # combine and run
    bodies = gal_A + gal_B

    run_sim(
        steps       = steps,
        bodies      = bodies,
        half_box    = half_box,
        theta       = theta,
        epsilon     = epsilon,
        dt          = dt,
        point_size  = point_sz,
        plot_type   = "density",
        fps=60
    )
