"""
Lightweight particle wrapper for the flat-quadtree Barnes–Hut solver.
"""

from __future__ import annotations

import numpy as np

# Gravitational constant in simulation units (kpc³ · kMs⁻¹ · (10 Myr)⁻²)
_G_KPC: np.float32 = np.float32(0.449)


def gravity() -> np.float32:
    """Return the gravitational constant *G* as float32."""
    return _G_KPC


class Body:
    """Minimal particle container with float32 state vectors.

    Args:
        mass: Particle mass.
        rx:   Initial *x* position.
        ry:   Initial *y* position.
        vx:   Initial *x* velocity.
        vy:   Initial *y* velocity.
        box_half: Half-box length *L* for periodic wrapping (optional).
        color: Matplotlib/PyQtGraph color identifier (for visualisers only).
    """

    __slots__ = ("m", "r", "v", "f", "box_half", "color")

    def __init__(
        self,
        mass: float,
        rx: float,
        ry: float,
        vx: float = 0.0,
        vy: float = 0.0,
        box_half: float = 0.0,
        color: str = "w",
    ) -> None:
        self.m: np.float32 = np.float32(mass)
        self.r: np.ndarray = np.asarray([rx, ry], dtype=np.float32)
        self.v: np.ndarray = np.asarray([vx, vy], dtype=np.float32)
        self.f: np.ndarray = np.zeros(2, dtype=np.float32)
        self.box_half: np.float32 = np.float32(box_half)
        self.color: str = color

    # --------------------------------------------------------------------- #
    # Low-level force bookkeeping                                           #
    # --------------------------------------------------------------------- #
    def reset_force(self) -> None:
        """Zero the accumulated force vector."""
        self.f[:] = 0.0

    def add_force(self, other: "Body", epsilon: float) -> None:
        """Add gravitational force contribution from *other* onto *self*.

        Args:
            other: The source body.
            epsilon: Softening length.
        """
        dr = self.r - other.r
        dist2 = dr.dot(dr) + epsilon * epsilon
        inv_d = np.float32(1.0) / np.sqrt(dist2)
        inv_d3 = inv_d / dist2
        factor = -gravity() * self.m * other.m * inv_d3
        self.f += dr * factor

    # --------------------------------------------------------------------- #
    # Time-integration helper (Euler–Cromer)                                #
    # --------------------------------------------------------------------- #
    def update(self, dt: float) -> None:
        """Advance *self* by `dt` using Euler–Cromer integration."""
        self.v += (self.f / self.m) * dt
        self.r = np.mod(self.r + self.v * dt + self.box_half, 2 * self.box_half) - self.box_half

    # --------------------------------------------------------------------- #
    # Array helpers for SoA ⇄ OO conversion                                 #
    # --------------------------------------------------------------------- #
    @staticmethod
    def to_arrays(bodies: list["Body"]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert a list of Body objects to SoA NumPy arrays.

        Returns:
            pos  – *(N, 2)* float32 positions
            vel  – *(N, 2)* float32 velocities
            mass – *(N,)*   float32 masses
        """
        n = len(bodies)
        pos = np.empty((n, 2), dtype=np.float32)
        vel = np.empty((n, 2), dtype=np.float32)
        mass = np.empty(n, dtype=np.float32)
        for i, b in enumerate(bodies):
            pos[i] = b.r
            vel[i] = b.v
            mass[i] = b.m
        return pos, vel, mass

    @staticmethod
    def from_arrays(bodies: list["Body"], pos: np.ndarray, vel: np.ndarray) -> None:
        """Write back *pos* and *vel* arrays into an existing Body list."""
        for i, b in enumerate(bodies):
            b.r[:] = pos[i]
            b.v[:] = vel[i]
