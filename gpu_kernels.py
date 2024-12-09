import numba as nb
import math
from numba import cuda

G = 1

@cuda.jit
def compute_all_forces_bh_gpu(d_positions, d_masses, d_nodes, theta, d_forces):

    softening=0.3

    i = cuda.grid(1)
    if i < d_positions.shape[0]:
        position = d_positions[i]
        mass = d_masses[i]
        force = cuda.local.array(2, dtype=nb.float64)
        force[0] = 0.0
        force[1] = 0.0
        stack = cuda.local.array(64, dtype=nb.int32)  # Adjust stack size as needed
        stack_ptr = 0
        stack[stack_ptr] = 0  # Start with root node
        stack_ptr += 1

        while stack_ptr > 0:
            stack_ptr -= 1
            node_index = stack[stack_ptr]
            node = d_nodes[node_index]

            if node['mass'] == 0 or node['body_index'] == i:
                continue

            dx = node['com_x'] - position[0]
            dy = node['com_y'] - position[1]
            r = math.sqrt(dx**2 + dy**2 + softening**2)

            width = node['xmax'] - node['xmin']

            if (width / r) < theta or node['body_index'] != -1:
                F = G * mass * node['mass'] / (r**3)
                force[0] += F * dx
                force[1] += F * dy
            else:
                if stack_ptr < len(stack):
                    if node['child0'] != -1:
                        stack[stack_ptr] = node['child0']
                        stack_ptr += 1
                    if node['child1'] != -1:
                        stack[stack_ptr] = node['child1']
                        stack_ptr += 1
                    if node['child2'] != -1:
                        stack[stack_ptr] = node['child2']
                        stack_ptr += 1
                    if node['child3'] != -1:
                        stack[stack_ptr] = node['child3']
                        stack_ptr += 1

        d_forces[i, 0] = force[0]
        d_forces[i, 1] = force[1]


@cuda.jit
def update_positions_and_velocities_gpu(d_positions, d_velocities, d_masses, d_forces, dt):
    i = cuda.grid(1)
    if i < d_positions.shape[0]:
        mass = d_masses[i]
        d_velocities[i, 0] += (d_forces[i, 0] / mass) * dt
        d_velocities[i, 1] += (d_forces[i, 1] / mass) * dt
        d_positions[i, 0] += d_velocities[i, 0] * dt
        d_positions[i, 1] += d_velocities[i, 1] * dt