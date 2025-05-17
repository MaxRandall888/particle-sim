# test_barnes_hut_sim.py
# test_barnes_hut_sim.py
"""
Extensive unit- and property-tests for the flat-quadtree Barnes–Hut solver.

Run with::

    pytest -q  # or  pytest --verbose
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Mock pyqtgraph and its Qt submodules
sys.modules['pyqtgraph'] = MagicMock()
sys.modules['pyqtgraph.Qt'] = MagicMock()
sys.modules['pyqtgraph.Qt.QtCore'] = MagicMock()
sys.modules['pyqtgraph.Qt.QtGui'] = MagicMock()
sys.modules['pyqtgraph.Qt.QtWidgets'] = MagicMock()

import math
import types
import copy

import numpy as np
import pytest

import array_quadtree as aq
import barnes_hut as bh
import body as bd
import utils as ut


# =============================================================================
# Helper utilities used by several tests
# =============================================================================
def direct_forces(pos: np.ndarray, mass: np.ndarray, eps: float) -> np.ndarray:
    """
    Reference O(N²) force calculator (float32) matching `Body.add_force`.
    """
    n = pos.shape[0]
    f = np.zeros_like(pos, dtype=np.float32)
    G = bd.gravity()
    for i in range(n):
        for j in range(i + 1, n):
            dr = pos[i] - pos[j]
            dist2 = np.dot(dr, dr) + eps * eps
            inv_d = 1.0 / math.sqrt(dist2)
            inv_d3 = inv_d / dist2
            fac = -G * mass[i] * mass[j] * inv_d3
            fij = dr * fac
            f[i] += fij
            f[j] -= fij
    return f


def naive_step(bodies: list[bd.Body], dt: float, eps: float) -> None:
    """
    One Euler-Cromer time step with the *exact* O(N²) algorithm.
    """
    for b in bodies:
        b.reset_force()

    n = len(bodies)
    for i in range(n):
        for j in range(i + 1, n):
            bodies[i].add_force(bodies[j], eps)
            bodies[j].add_force(bodies[i], eps)

    for b in bodies:
        b.update(dt)


# =============================================================================
# Morton helpers & bit–interleaving
# =============================================================================
def test_interleave_bits_basic():
    """
    A few exhaustively-known values for `_interleave_bits`.
    """
    # (x, y) → expected Morton code
    table = {
        (0, 0): 0,
        (1, 0): 1,
        (0, 1): 2,       # binary 10
        (1, 1): 3,       # binary 11
        (2, 3): 0b1110,  # hand-worked small example corrected (decimal 14)
    }
    for (x, y), ref in table.items():
        code = aq._interleave_bits(np.uint32(x), np.uint32(y))
        assert int(code) == ref, f"({x},{y}) produced {code}, expected {ref}"



def test_morton_codes_monotone():
    """
    Morton codes should preserve a weak ordering along each axis.
    """
    half_box = 1.0
    pts = np.array([[-half_box, -half_box],
                    [0.0,       0.0],
                    [half_box,  half_box]], dtype=np.float32)
    codes = aq.morton_codes(pts, half_box)
    # Codes must increase for the chosen diagonal.
    assert np.all(np.diff(codes.astype(np.int64)) > 0)


# =============================================================================
# FlatQuadTree build / force symmetry & accuracy
# =============================================================================
@pytest.mark.parametrize("theta", [0.4, 0.1])
def test_flatquadtree_force_symmetry(theta):
    """
    For two equal-mass bodies the forces must be equal and opposite.
    """
    pos = np.array([[0.0, 0.0],
                    [1.0, 0.0]], dtype=np.float32)
    mass = np.array([1.0, 1.0], dtype=np.float32)
    tree = aq.FlatQuadTree(half_box=2.0, max_bodies=2)
    tree.build(pos, mass)
    F = tree.forces(pos, mass, theta=theta, eps=1e-6)

    # Opposite sign, equal magnitude
    np.testing.assert_allclose(F[0], -F[1], rtol=1e-5, atol=1e-7)


def test_flatquadtree_matches_direct_small_system():
    """
    With a *very* small θ the Barnes–Hut result should match direct forces.
    """
    rng = np.random.default_rng(123)
    n = 10
    pos = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)
    mass = rng.uniform(0.1, 2.0, size=n).astype(np.float32)

    tree = aq.FlatQuadTree(half_box=1.5, max_bodies=n)
    tree.build(pos, mass)
    F_bh = tree.forces(pos, mass, theta=1e-4, eps=0.0)

    F_ref = direct_forces(pos, mass, eps=0.0)

    np.testing.assert_allclose(F_bh, F_ref, rtol=1e-3, atol=1e-5)


# =============================================================================
# Body class – bookkeeping helpers
# =============================================================================
def test_body_force_addition_and_reset():
    b1 = bd.Body(1.0, 0.0, 0.0)
    b2 = bd.Body(2.0, 1.0, 0.0)
    assert np.all(b1.f == 0)
    b1.add_force(b2, epsilon=0.0)
    assert not np.all(b1.f == 0), "Force should have been accumulated"
    b1.reset_force()
    assert np.allclose(b1.f, 0), "reset_force failed"


def test_body_update_periodic_wrap():
    """
    A body crossing +L must re-enter at −L (periodic boundary).
    """
    L = 1.0
    b = bd.Body(1.0, 0.99, 0.0, vx=0.2, vy=0.0, box_half=L)
    b.update(dt=1.0)
    assert -L <= b.r[0] <= L
    # Since vx>0, the x-coordinate should now be around −0.81
    assert b.r[0] < 0.0


def test_to_from_arrays_roundtrip():
    bodies = [bd.Body(1.0, 0.1 * i, 0.2 * i, vx=0.01, vy=-0.02) for i in range(5)]
    pos0, vel0, m0 = bd.Body.to_arrays(bodies)
    # Modify arrays and write back
    pos0 += 0.5
    vel0 *= -1
    bd.Body.from_arrays(bodies, pos0, vel0)
    for i, b in enumerate(bodies):
        np.testing.assert_array_equal(b.r, pos0[i])
        np.testing.assert_array_equal(b.v, vel0[i])
        assert b.m == m0[i]


# =============================================================================
# bh_step – correctness vs. direct algorithm (3-body system)
# =============================================================================
def _clone_bodies(bodies: list[bd.Body]) -> list[bd.Body]:
    """
    Deep-copy a list of Body objects (stateful).
    """
    return [bd.Body(b.m, *b.r, *b.v, box_half=b.box_half) for b in bodies]


def test_bh_step_matches_naive_three_body():
    """
    Run a few steps for 3 bodies and compare BH (θ very small) to the
    exact O(N²) algorithm.
    """
    # Triangle configuration
    bodies0 = [
        bd.Body(1.0, -0.5, 0.0, 0.0,  0.1, box_half=5.0),
        bd.Body(1.0,  0.5, 0.0, 0.0, -0.1, box_half=5.0),
        bd.Body(1.0,  0.0, 0.8, 0.0,  0.0, box_half=5.0),
    ]
    bh_bodies = _clone_bodies(bodies0)
    ref_bodies = _clone_bodies(bodies0)

    dt = 0.02
    eps = 0.0
    theta = 1e-4     # practically exact

    tree_buf = None
    for _ in range(10):                   # ten small steps
        # Barnes–Hut
        bh_bodies, tree_buf = ut.bh_step(
            bh_bodies, half_box=5.0, theta=theta,
            epsilon=eps, dt=dt, tree_buffer=tree_buf
        )
        # Direct
        naive_step(ref_bodies, dt=dt, eps=eps)

    pos_bh, vel_bh, _ = bd.Body.to_arrays(bh_bodies)
    pos_ref, vel_ref, _ = bd.Body.to_arrays(ref_bodies)

    np.testing.assert_allclose(pos_bh, pos_ref, rtol=2e-3, atol=5e-4)
    np.testing.assert_allclose(vel_bh, vel_ref, rtol=2e-3, atol=5e-4)


# =============================================================================
# Particle “culling” (periodic wrap) – stays in bounds after many steps
# =============================================================================
def test_particles_stay_within_domain_after_many_steps():
    """
    After multiple updates no coordinate should exceed ±half_box.
    """
    half_box = 2.0
    rng = np.random.default_rng(42)
    bodies = [
        bd.Body(1.0,
                rng.uniform(-half_box, half_box),
                rng.uniform(-half_box, half_box),
                *rng.uniform(-0.5, 0.5, size=2),
                box_half=half_box)
        for _ in range(20)
    ]
    dt = 0.1
    for _ in range(200):  # 20 time units
        for b in bodies:
            b.update(dt)
            assert -half_box <= b.r[0] <= half_box
            assert -half_box <= b.r[1] <= half_box
