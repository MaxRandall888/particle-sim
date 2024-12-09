import numpy as np
import numpy as np
from numba import cuda

from body import initialize_bodies
from quadtree import build_quadtree
from gpu_kernels import compute_all_forces_bh_gpu, update_positions_and_velocities_gpu

# Gravitational constant (arbitrary units)
G = 1

# Function to calculate the velocity for a stable circular orbit
def calculate_orbital_velocity(r, mass_central):
    return np.sqrt(G * mass_central / r)

def run_simulation():
    num_bodies = 1000000
    mass_particle = 1
    particle_size = 1  # Adjusted for PyQtGraph
    particle_color = 'w'  # 'w' for white in PyQtGraph
    mass_central = 1
    max_radius = 50
    dt = 0.005
    theta = 0.5
    max_nodes = 10000000

    bodies, visual_bodies = initialize_bodies(
        num_bodies, max_radius, mass_central, mass_particle, particle_size, particle_color
    )
    positions = np.ascontiguousarray(bodies['position'])
    velocities = np.ascontiguousarray(bodies['velocity'])
    masses = np.ascontiguousarray(bodies['mass'])

    d_positions = cuda.to_device(positions)
    d_velocities = cuda.to_device(velocities)
    d_masses = cuda.to_device(masses)

    threads_per_block = 256
    blocks_per_grid = (num_bodies + threads_per_block - 1) // threads_per_block

    # PyQtGraph setup
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore

    # Set global configuration options
    pg.setConfigOptions(useOpenGL=True, background='k')  # Set background color to black

    app = QtWidgets.QApplication([])

    # Create a window with a GraphicsLayout
    win = pg.GraphicsLayoutWidget(show=True, title="Particle Simulation")
    plot = win.addPlot()
    plot.setAspectLocked(True)
    plot.setXRange(-max_radius * 2, max_radius * 2)
    plot.setYRange(-max_radius * 2, max_radius * 2)
    plot.hideAxis('bottom')
    plot.hideAxis('left')

    # Create a scatter plot item
    scatter = pg.ScatterPlotItem(size=particle_size, pen=None, brush=particle_color)
    plot.addItem(scatter)

    # Timer for updating the simulation
    timer = QtCore.QTimer()

    def update():
        nonlocal positions, velocities, masses, d_positions, d_velocities, d_masses

        nodes = build_quadtree(positions, masses, max_nodes)
        d_nodes = cuda.to_device(nodes)
        d_forces = cuda.device_array_like(positions)

        compute_all_forces_bh_gpu[blocks_per_grid, threads_per_block](
            d_positions, d_masses, d_nodes, theta, d_forces
        )
        update_positions_and_velocities_gpu[blocks_per_grid, threads_per_block](
            d_positions, d_velocities, d_masses, d_forces, dt
        )

        positions = d_positions.copy_to_host()

        # Update scatter plot data
        scatter.setData(positions[:, 0], positions[:, 1])

    timer.timeout.connect(update)
    timer.start(0)  # Run as fast as possible; adjust interval if needed

    QtWidgets.QApplication.instance().exec_()

if __name__ == "__main__":
    run_simulation()