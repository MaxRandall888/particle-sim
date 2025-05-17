"""
Real-time driver for the array-based Barnes–Hut simulation with PyQtGraph.
"""

from __future__ import annotations

import numpy as np
from body import Body
from utils import run_sim, generate_galaxy, generate_random_grid

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
