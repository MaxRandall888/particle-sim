"""
array_quadtree.py
=================
Flat (array-backed) quadtree with Numba-accelerated kernels.

New in this revision
--------------------
1. **Bounding-box containment test** – we now store each node’s geometric
   centre (*gx*, *gy*) and only subtract a particle’s own mass when the node
   actually contains that particle.
2. **Zero-distance guard** – nodes whose centre lies exactly on a particle are
   skipped, preventing NaNs.
3. **Float64 centre-of-mass and mass arrays** – improves numerical precision
   for very massive cores embedded in light haloes.
"""

from __future__ import annotations
import numpy as np
from numba import njit, prange
from body import gravity

# --------------------------------------------------------------------------- #
# Morton utilities                                                            #
# --------------------------------------------------------------------------- #
@njit
def _interleave_bits(x: np.uint32, y: np.uint32) -> np.uint32:
    """Interleave the lower 16 bits of *x* and *y* (Morton/Z-order code).

    Args:
        x: Lower 16 bits become even bits.
        y: Lower 16 bits become odd bits.

    Returns:
        32-bit Morton code.
    """
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    y = (y | (y << 8)) & 0x00FF00FF
    y = (y | (y << 4)) & 0x0F0F0F0F
    y = (y | (y << 2)) & 0x33333333
    y = (y | (y << 1)) & 0x55555555
    return x | (y << 1)


@njit
def morton_codes(pos: np.ndarray, half_box: float) -> np.ndarray:
    """Return Morton codes for *pos* ∈ ``[-half_box, +half_box]``.

    Args:
        pos: *(N, 2)* positions.
        half_box: Half the simulation-box width.

    Returns:
        *(N,)* `np.uint32` array of codes.
    """
    n = pos.shape[0]
    codes = np.empty(n, dtype=np.uint32)
    scale = 65535.0 / (2.0 * half_box)
    for i in range(n):
        xi = int((pos[i, 0] + half_box) * scale)
        yi = int((pos[i, 1] + half_box) * scale)
        xi = 0 if xi < 0 else 65535 if xi > 65535 else xi
        yi = 0 if yi < 0 else 65535 if yi > 65535 else yi
        codes[i] = _interleave_bits(np.uint32(xi), np.uint32(yi))
    return codes


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #
@njit
def _init_node(
    idx: int,
    cx: np.ndarray,
    cy: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    mass: np.ndarray,
    half: np.ndarray,
    child: np.ndarray,
    leaf: np.ndarray,
    gx0: float,
    gy0: float,
    m: float,
    hs: float,
) -> None:
    """Initialise a node slot.

    Args:
        idx: Index of node to initialise.
        cx, cy: Centre-of-mass arrays (float64).
        gx, gy: Geometric-centre arrays (float32).
        mass: Total-mass array (float64).
        half: Half-side array.
        child: Child-index array *(M, 4)*.
        leaf: Byte array, 1 = leaf.
        gx0, gy0: Geometric centre coordinates.
        m: Initial mass.
        hs: Half-side length of square.
    """
    cx[idx] = 0.0
    cy[idx] = 0.0
    gx[idx] = gx0
    gy[idx] = gy0
    mass[idx] = m
    half[idx] = hs
    leaf[idx] = 1
    for k in range(4):
        child[idx, k] = -1


@njit
def _child_slot(px: float, py: float, gx0: float, gy0: float) -> int:
    """Quadrant of point *(px,py)* relative to square centre *(gx0,gy0)*."""
    east = 1 if px >= gx0 else 0
    north = 1 if py >= gy0 else 0
    return north * 2 + east


# --------------------------------------------------------------------------- #
# Quadtree build                                                               #
# --------------------------------------------------------------------------- #
@njit
def build_tree(
    pos: np.ndarray,
    mass_b: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    mass_n: np.ndarray,
    half: np.ndarray,
    child: np.ndarray,
    leaf: np.ndarray,
    half_box: float,
) -> int:
    """Breadth-first quadtree construction.

    Args:
        pos, mass_b: Particle arrays.
        cx, cy, gx, gy, mass_n, half, child, leaf: Node arrays.
        half_box: Half-width of root node.

    Returns:
        Number of nodes created.
    """
    n_bodies = pos.shape[0]
    _init_node(
        0, cx, cy, gx, gy, mass_n, half, child, leaf,
        0.0, 0.0, 0.0, half_box
    )
    next_free = 1

    for i in range(n_bodies):
        px, py, pm = pos[i, 0], pos[i, 1], mass_b[i]
        idx = 0
        while True:
            m_old = mass_n[idx]
            new_m = m_old + pm
            if m_old == 0.0:
                cx[idx], cy[idx] = px, py
            else:
                cx[idx] = (cx[idx] * m_old + px * pm) / new_m
                cy[idx] = (cy[idx] * m_old + py * pm) / new_m
            mass_n[idx] = new_m

            if leaf[idx] == 1:
                if child[idx, 0] == -1:              # empty leaf
                    child[idx, 0] = i
                    break

                # subdivide
                b_idx = child[idx, 0]
                child[idx, 0] = -1
                leaf[idx] = 0
                new_half = half[idx] * 0.5
                for k in range(4):
                    offset_x = -new_half if k & 1 == 0 else new_half
                    offset_y = -new_half if k < 2      else new_half
                    _init_node(
                        next_free + k,
                        cx, cy, gx, gy, mass_n, half, child, leaf,
                        gx[idx] + offset_x, gy[idx] + offset_y,
                        0.0, new_half
                    )
                    child[idx, k] = next_free + k
                next_free += 4

                # re-insert stored body
                px, py, pm = pos[b_idx, 0], pos[b_idx, 1], mass_b[b_idx]
                idx = child[idx, _child_slot(px, py, gx[idx], gy[idx])]
            else:
                idx = child[idx, _child_slot(px, py, gx[idx], gy[idx])]

    return next_free


# --------------------------------------------------------------------------- #
# Force kernel                                                                 #
# --------------------------------------------------------------------------- #
@njit(parallel=True, fastmath=True)
def compute_forces(
    pos: np.ndarray,
    mass_b: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    mass_n: np.ndarray,
    half: np.ndarray,
    child: np.ndarray,
    leaf: np.ndarray,
    num_nodes: int,
    theta: float,
    eps: float,
    G: float = gravity(),
) -> np.ndarray:
    """Barnes–Hut force computation.

    Args:
        pos, mass_b: Particle data.
        cx, cy, gx, gy, mass_n, half, child, leaf: Node arrays.
        num_nodes: Number of valid nodes in arrays.
        theta: Opening-angle criterion.
        eps: Plummer softening length.
        G: Gravitational constant.

    Returns:
        *(N, 2)* force array (float32).
    """
    n = pos.shape[0]
    forces = np.zeros((n, 2), dtype=np.float32)

    for i in prange(n):
        fx = fy = 0.0
        stack = np.empty(64, dtype=np.int32)
        sp = 1
        stack[0] = 0

        while sp > 0:
            sp -= 1
            idx = stack[sp]

            dx = cx[idx] - pos[i, 0]
            dy = cy[idx] - pos[i, 1]
            dist2 = dx*dx + dy*dy + eps*eps
            if dist2 == 0.0:                      # exact overlap
                continue
            dist = np.sqrt(dist2)

            if leaf[idx] == 1 and child[idx, 0] == i:
                continue                          # same particle

            # containment test for self-mass subtraction
            contains_self = (
                (pos[i, 0] >= gx[idx] - half[idx]) and
                (pos[i, 0] <= gx[idx] + half[idx]) and
                (pos[i, 1] >= gy[idx] - half[idx]) and
                (pos[i, 1] <= gy[idx] + half[idx])
            )

            if leaf[idx] == 1 or (2.0 * half[idx]) / dist <= theta:
                inv_d3 = 1.0 / (dist * dist2)
                m_eff = mass_n[idx] - mass_b[i] if contains_self else mass_n[idx]
                if m_eff <= 0.0:
                    continue
                f = G * mass_b[i] * m_eff * inv_d3
                fx += dx * f
                fy += dy * f
            else:
                for k in range(4):
                    cidx = child[idx, k]
                    if cidx != -1:
                        stack[sp] = cidx
                        sp += 1

        forces[i, 0] = fx
        forces[i, 1] = fy

    return forces


# --------------------------------------------------------------------------- #
# Public façade                                                               #
# --------------------------------------------------------------------------- #
class FlatQuadTree:
    """High-level wrapper around the JIT kernels."""

    def __init__(self, half_box: float, max_bodies: int) -> None:
        """
        Args:
            half_box: Half the simulation-box width.
            max_bodies: Upper bound on number of particles.
        """
        self.half_box = np.float32(half_box)
        max_nodes = max_bodies * 4

        # float64 for COM and mass, float32 elsewhere
        self.cx   = np.empty(max_nodes, dtype=np.float64)
        self.cy   = np.empty(max_nodes, dtype=np.float64)
        self.gx   = np.empty(max_nodes, dtype=np.float32)
        self.gy   = np.empty(max_nodes, dtype=np.float32)
        self.mass = np.empty(max_nodes, dtype=np.float64)
        self.half = np.empty(max_nodes, dtype=np.float32)
        self.child = np.empty((max_nodes, 4), dtype=np.int32)
        self.leaf  = np.empty(max_nodes, dtype=np.uint8)
        self.num_nodes: int = 0

    # --------------------------------------------------------------------- #
    # Build                                                                  #
    # --------------------------------------------------------------------- #
    def build(self, pos: np.ndarray, mass_b: np.ndarray) -> None:
        """Construct the tree from arrays.

        Args:
            pos: *(N, 2)* positions.
            mass_b: *(N,)* masses.
        """
        self.num_nodes = build_tree(
            pos.astype(np.float32, copy=False),
            mass_b.astype(np.float32, copy=False),
            self.cx, self.cy, self.gx, self.gy,
            self.mass, self.half, self.child, self.leaf,
            self.half_box,
        )

    # --------------------------------------------------------------------- #
    # Forces                                                                 #
    # --------------------------------------------------------------------- #
    def forces(
        self,
        pos: np.ndarray,
        mass_b: np.ndarray,
        theta: float = 0.5,
        eps: float = 0.01,
    ) -> np.ndarray:
        """Compute particle forces.

        Args:
            pos: *(N, 2)* positions.
            mass_b: *(N,)* masses.
            theta: Opening-angle.
            eps: Softening length.

        Returns:
            *(N, 2)* forces.
        """
        return compute_forces(
            pos.astype(np.float32, copy=False),
            mass_b.astype(np.float32, copy=False),
            self.cx, self.cy, self.gx, self.gy,
            self.mass, self.half, self.child, self.leaf,
            self.num_nodes,
            np.float32(theta), np.float32(eps),
        )
