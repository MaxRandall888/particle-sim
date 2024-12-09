import numpy as np

# Numba-compatible structured array for bodies
body_dtype_numba = np.dtype([
    ('position', np.float64, (2,)),
    ('velocity', np.float64, (2,)),
    ('mass', np.float64)
])

# Array for visualization
visual_dtype = np.dtype([
    ('size', np.float64),
    ('color', 'U10')
])

# Gravitational constant (arbitrary units)
G = 1

# Function to calculate the velocity for a stable circular orbit
def calculate_orbital_velocity(r, mass_central):
    return np.sqrt(G * mass_central / r)

def initialize_bodies(num_bodies, max_radius, mass_central, particle_mass, particle_size, particle_color):
    total_particle_mass = num_bodies * particle_mass

    bodies = np.zeros(num_bodies + 1, dtype=body_dtype_numba)
    visual_bodies = np.zeros(num_bodies + 1, dtype=visual_dtype)

    radii = np.random.uniform(max_radius * 0.1, max_radius, num_bodies)
    angles = np.random.uniform(0, 2 * np.pi, num_bodies)
    sorted_indices = np.argsort(radii)
    radii = radii[sorted_indices]
    angles = angles[sorted_indices]

    for i in range(num_bodies):
        r = radii[i]
        theta = angles[i]
        x, y = r * np.cos(theta), r * np.sin(theta)

        area_fraction = (r / max_radius)**2
        mass_enclosed = mass_central + total_particle_mass * area_fraction

        v = calculate_orbital_velocity(r, mass_enclosed)
        vx, vy = -v * np.sin(theta), v * np.cos(theta)

        bodies[i]['position'] = [x, y]
        bodies[i]['velocity'] = [vx, vy]
        bodies[i]['mass'] = particle_mass

        visual_bodies[i]['size'] = particle_size
        visual_bodies[i]['color'] = particle_color

    bodies[num_bodies]['position'] = [0.0, 0.0]
    bodies[num_bodies]['velocity'] = [0.0, 0.0]
    bodies[num_bodies]['mass'] = mass_central

    visual_bodies[num_bodies]['size'] = particle_size
    visual_bodies[num_bodies]['color'] = particle_color

    return bodies, visual_bodies
