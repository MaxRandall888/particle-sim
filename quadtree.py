import numpy as np
import numba as nb

# Define structured array for Quadtree Nodes
node_dtype = np.dtype([
    ('xmin', np.float64),
    ('xmax', np.float64),
    ('ymin', np.float64),
    ('ymax', np.float64),
    ('mass', np.float64),
    ('com_x', np.float64),
    ('com_y', np.float64),
    ('body_index', np.int64),
    ('child0', np.int64),
    ('child1', np.int64),
    ('child2', np.int64),
    ('child3', np.int64)
]).newbyteorder('<')

@nb.njit
def get_child_index(nodes, node_index, position):
    x = position[0]
    y = position[1]
    xmid = 0.5 * (nodes[node_index]['xmin'] + nodes[node_index]['xmax'])
    ymid = 0.5 * (nodes[node_index]['ymin'] + nodes[node_index]['ymax'])
    
    if x <= xmid:
        if y <= ymid:
            return nodes[node_index]['child0']  # SW
        else:
            return nodes[node_index]['child1']  # NW
    else:
        if y <= ymid:
            return nodes[node_index]['child2']  # SE
        else:
            return nodes[node_index]['child3']  # NE

@nb.njit
def set_child_index(nodes, node_index, position, child_node_index):
    x = position[0]
    y = position[1]
    xmid = 0.5 * (nodes[node_index]['xmin'] + nodes[node_index]['xmax'])
    ymid = 0.5 * (nodes[node_index]['ymin'] + nodes[node_index]['ymax'])
    
    if x <= xmid:
        if y <= ymid:
            nodes[node_index]['child0'] = child_node_index  # SW
        else:
            nodes[node_index]['child1'] = child_node_index  # NW
    else:
        if y <= ymid:
            nodes[node_index]['child2'] = child_node_index  # SE
        else:
            nodes[node_index]['child3'] = child_node_index  # NE

@nb.njit
def subdivide_node(nodes, node_index, node_count):
    # (No changes needed here)
    xmin = nodes[node_index]['xmin']
    xmax = nodes[node_index]['xmax']
    ymin = nodes[node_index]['ymin']
    ymax = nodes[node_index]['ymax']
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    
    # Create 4 child nodes
    for i in range(4):
        if node_count[0] >= nodes.shape[0]:
            # Resize nodes array if necessary (not implemented here)
            continue  # For simplicity, skip if max_nodes reached
        child_index = node_count[0]
        node_count[0] += 1
        
        # Initialize child node
        nodes[child_index]['mass'] = 0.0
        nodes[child_index]['com_x'] = 0.0
        nodes[child_index]['com_y'] = 0.0
        nodes[child_index]['body_index'] = -1
        nodes[child_index]['child0'] = -1
        nodes[child_index]['child1'] = -1
        nodes[child_index]['child2'] = -1
        nodes[child_index]['child3'] = -1
        
        if i == 0:
            # SW quadrant
            nodes[child_index]['xmin'] = xmin
            nodes[child_index]['xmax'] = xmid
            nodes[child_index]['ymin'] = ymin
            nodes[child_index]['ymax'] = ymid
            nodes[node_index]['child0'] = child_index
        elif i == 1:
            # NW quadrant
            nodes[child_index]['xmin'] = xmin
            nodes[child_index]['xmax'] = xmid
            nodes[child_index]['ymin'] = ymid
            nodes[child_index]['ymax'] = ymax
            nodes[node_index]['child1'] = child_index
        elif i == 2:
            # SE quadrant
            nodes[child_index]['xmin'] = xmid
            nodes[child_index]['xmax'] = xmax
            nodes[child_index]['ymin'] = ymin
            nodes[child_index]['ymax'] = ymid
            nodes[node_index]['child2'] = child_index
        else:
            # NE quadrant
            nodes[child_index]['xmin'] = xmid
            nodes[child_index]['xmax'] = xmax
            nodes[child_index]['ymin'] = ymid
            nodes[child_index]['ymax'] = ymax
            nodes[node_index]['child3'] = child_index

@nb.njit
def insert_body(nodes, node_index, positions, masses, body_index, node_count):
    # Check if node is a leaf
    if nodes[node_index]['body_index'] == -1 and nodes[node_index]['child0'] == -1:
        # Empty leaf node, insert body here
        nodes[node_index]['body_index'] = body_index
        nodes[node_index]['mass'] = masses[body_index]
        nodes[node_index]['com_x'] = positions[body_index][0]
        nodes[node_index]['com_y'] = positions[body_index][1]
    else:
        # Internal node or occupied leaf node
        if nodes[node_index]['child0'] == -1:
            # Subdivide the node
            subdivide_node(nodes, node_index, node_count)
        
        # Update mass and center of mass
        m1 = nodes[node_index]['mass']
        m2 = masses[body_index]
        com1_x = nodes[node_index]['com_x']
        com1_y = nodes[node_index]['com_y']
        com2_x = positions[body_index][0]
        com2_y = positions[body_index][1]
        total_mass = m1 + m2
        nodes[node_index]['mass'] = total_mass
        nodes[node_index]['com_x'] = (com1_x * m1 + com2_x * m2) / total_mass
        nodes[node_index]['com_y'] = (com1_y * m1 + com2_y * m2) / total_mass
        
        # Insert the existing body into a child
        existing_body_index = nodes[node_index]['body_index']
        if existing_body_index != -1:
            nodes[node_index]['body_index'] = -1
            child_index = get_child_index(nodes, node_index, positions[existing_body_index])
            if child_index == -1:
                # Create child node
                child_node_index = node_count[0]
                node_count[0] += 1
                nodes[child_node_index]['mass'] = 0.0
                nodes[child_node_index]['com_x'] = 0.0
                nodes[child_node_index]['com_y'] = 0.0
                nodes[child_node_index]['body_index'] = -1
                nodes[child_node_index]['child0'] = -1
                nodes[child_node_index]['child1'] = -1
                nodes[child_node_index]['child2'] = -1
                nodes[child_node_index]['child3'] = -1
                set_child_index(nodes, node_index, positions[existing_body_index], child_node_index)
                insert_body(nodes, child_node_index, positions, masses, existing_body_index, node_count)
            else:
                insert_body(nodes, child_index, positions, masses, existing_body_index, node_count)
        
        # Insert the new body into a child
        child_index = get_child_index(nodes, node_index, positions[body_index])
        if child_index == -1:
            # Create child node
            child_node_index = node_count[0]
            node_count[0] += 1
            nodes[child_node_index]['mass'] = 0.0
            nodes[child_node_index]['com_x'] = 0.0
            nodes[child_node_index]['com_y'] = 0.0
            nodes[child_node_index]['body_index'] = -1
            nodes[child_node_index]['child0'] = -1
            nodes[child_node_index]['child1'] = -1
            nodes[child_node_index]['child2'] = -1
            nodes[child_node_index]['child3'] = -1
            set_child_index(nodes, node_index, positions[body_index], child_node_index)
            insert_body(nodes, child_node_index, positions, masses, body_index, node_count)
        else:
            insert_body(nodes, child_index, positions, masses, body_index, node_count)

@nb.njit
def build_quadtree(positions, masses, max_nodes):
    num_bodies = positions.shape[0]
    nodes = np.zeros(max_nodes, dtype=node_dtype)
    node_count = np.array([1])  # Start with 1 node (the root)
    
    # Initialize root node
    xmin = np.min(positions[:, 0])
    xmax = np.max(positions[:, 0])
    ymin = np.min(positions[:, 1])
    ymax = np.max(positions[:, 1])
    
    nodes[0]['xmin'] = xmin
    nodes[0]['xmax'] = xmax
    nodes[0]['ymin'] = ymin
    nodes[0]['ymax'] = ymax
    nodes[0]['mass'] = 0.0
    nodes[0]['com_x'] = 0.0
    nodes[0]['com_y'] = 0.0
    nodes[0]['body_index'] = -1
    nodes[0]['child0'] = -1
    nodes[0]['child1'] = -1
    nodes[0]['child2'] = -1
    nodes[0]['child3'] = -1
    
    # Insert bodies into the tree
    for i in range(num_bodies):
        insert_body(nodes, 0, positions, masses, i, node_count)
    
    return nodes