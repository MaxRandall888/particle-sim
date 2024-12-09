import pytest
import numpy as np
from numba import cuda
from gpu_kernels import compute_all_forces_bh_gpu

# Gravitational constant and softening factor for testing
G = 1.0
softening = 0.1

def create_mock_data():
    """
    Create mock data for positions, masses, and quadtree nodes.
    """
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    masses = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    nodes = np.zeros(4, dtype=[
        ('xmin', np.float64), ('xmax', np.float64),
        ('ymin', np.float64), ('ymax', np.float64),
        ('mass', np.float64), ('com_x', np.float64),
        ('com_y', np.float64), ('body_index', np.int64),
        ('child0', np.int64), ('child1', np.int64),
        ('child2', np.int64), ('child3', np.int64)
    ])
    nodes[0]['xmin'] = -1.0
    nodes[0]['xmax'] = 2.0
    nodes[0]['ymin'] = -1.0
    nodes[0]['ymax'] = 2.0
    nodes[0]['mass'] = 3.0
    nodes[0]['com_x'] = 1.0 / 3.0
    nodes[0]['com_y'] = 1.0 / 3.0
    return positions, masses, nodes

def compute_force_manually(positions, masses, softening=1):
    """
    Manually compute the gravitational forces for validation.
    """
    num_bodies = positions.shape[0]
    forces = np.zeros_like(positions)

    for i in range(num_bodies):
        for j in range(num_bodies):
            if i != j:
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                r = (dx**2 + dy**2 + softening)**0.5
                f = G * masses[i] * masses[j] / r**2
                forces[i, 0] += f * dx / r
                forces[i, 1] += f * dy / r
    return forces

def test_compute_all_forces_bh_gpu():
    """
    Test the CUDA kernel `compute_all_forces_bh_gpu` for proportionality and correctness.
    """
    cuda.select_device(0)  # Ensure a CUDA context is initialized

    # Generate mock data
    positions, masses, nodes = create_mock_data()

    # Print initial data for debugging
    print("Initial Positions:", positions)
    print("Initial Masses:", masses)
    print("Initial Nodes:", nodes)

    # Transfer data to GPU
    d_positions = cuda.to_device(positions)
    d_masses = cuda.to_device(masses)
    d_nodes = cuda.to_device(nodes)
    d_forces = cuda.device_array_like(positions)

    # Ensure same initial conditions for both GPU and manual calculations
    initial_positions = positions.copy()
    initial_masses = masses.copy()

    # Define kernel launch parameters
    threads_per_block = 256
    blocks_per_grid = (positions.shape[0] + threads_per_block - 1) // threads_per_block

    # Launch kernel
    compute_all_forces_bh_gpu[blocks_per_grid, threads_per_block](
        d_positions, d_masses, d_nodes, 0.5, d_forces
    )
    cuda.synchronize()

    # Copy forces from GPU to host
    gpu_forces = d_forces.copy_to_host()
    print("Computed Forces (GPU):", gpu_forces)

    # Check input data on GPU after kernel execution
    gpu_positions = d_positions.copy_to_host()
    gpu_masses = d_masses.copy_to_host()
    print("Positions on GPU after kernel:", gpu_positions)
    print("Masses on GPU after kernel:", gpu_masses)

    # Compute forces manually using the same initial conditions
    manual_forces = compute_force_manually(initial_positions, initial_masses, softening=softening)
    print("Computed Forces (Manual):", manual_forces)

    # Validate GPU forces against manual forces
    assert np.allclose(
        gpu_forces, manual_forces, atol=1e-2
    ), f"GPU and Manual forces mismatch: GPU={gpu_forces} vs Manual={manual_forces}"

    print("Forces match within tolerance!")

