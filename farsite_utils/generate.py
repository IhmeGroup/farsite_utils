"""Generate random 2D fields."""

import math
import numpy as np
from shapely import geometry


def randomUniform(field_shape, low, high, dtype=None):
    """Generate field with uniformly-distrubuted noise."""

    return np.random.uniform(low=low, high=high, size=field_shape).astype(dtype)


def randomInteger(field_shape, low, high, dtype=np.int64):
    """Generate field with uniformly-distributed integer noise."""

    return np.random.randint(low=low, high=high, size=field_shape, dtype=dtype)


def randomNormal(field_shape, mean=0, stddev=0.25, dtype=None):
    """Generate field with normally-distrubuted noise."""

    return np.random.normal(loc=mean, scale=stddev, size=field_shape)


def randomFoldedNormal(field_shape, mean=1, stddev=0.25, dtype=None):
    """Generate field with folded normally-distributed noise."""

    return np.abs(randomNormal(field_shape, mean, stddev, dtype))


def randomBool(field_shape, p=0.5, dtype=bool):
    """Generate field with boolean values, given probability of True."""

    raw = np.random.rand(*field_shape)
    field = np.less_equal(raw, p)
    return field.astype(dtype)


def randomChoice(field_shape, vals, dtype=None):
    """Generate field selecting random value from list of given vals."""

    return np.random.choice(vals, field_shape).astype(dtype)


def randomPatchy(field_shape, vals, base, p_filled, patch_sides, patch_radius_mean, patch_radius_stdev=0, dtype=None):
    """Generate patchy field."""

    A_field = field_shape[0] * field_shape[1]

    field = (np.zeros(field_shape) + base).astype(dtype)
    field_bool = np.zeros(field_shape, dtype=bool)

    p_actual = 0
    while p_actual < p_filled:

        # Choose fuel model to apply
        val_current = np.random.choice(vals)

        # Generate patch
        radius = np.random.normal(loc=patch_radius_mean, scale=patch_radius_stdev)
        rotation = np.random.uniform(0.0, 360.0)
        translation = (
            np.random.uniform(0, field_shape[0]),
            np.random.uniform(0, field_shape[1]))
        patch = regularPolygon(patch_sides, radius, rotation, translation)

        # Restrict search bounds to bounding box of patch in field
        search_bounds_x = (
            int(np.max((np.floor(patch.bounds[0]),  0))),
            int(np.min((np.ceil (patch.bounds[2]), field_shape[0]))))
        search_bounds_y = (
            int(np.max((np.floor(patch.bounds[1]), 0))),
            int(np.min((np.ceil (patch.bounds[3]), field_shape[1]))))

        # Apply fuel model in region covered by patch
        for i in range(*search_bounds_x):
            for j in range(*search_bounds_y):
                if patch.contains(geometry.Point(i, j)):
                    field[i,j] = val_current
                    field_bool[i,j] = True
        
        # Check covered area
        p_actual = np.count_nonzero(field_bool) / A_field
    
    return field


def gradient(field_shape, aspect, slope, length_scale=1.0, dtype=None):
    """Generate field with constant given slope in given direction.
    Aspect refers to DOWN-SLOPE direction, measured CLOCKWISE from NORTH.
    All angles in radians."""

    plane_norm = np.array([np.sin(slope) * np.cos(aspect + np.pi/2),
                           np.sin(slope) * np.sin(aspect + np.pi/2),
                           np.cos(slope)])
    
    x = np.arange(field_shape[0])
    y = np.arange(field_shape[1])
    [X, Y] = np.meshgrid(x, y)

    field = length_scale * (plane_norm[0]*X + plane_norm[1]*Y) / plane_norm[2]
    field -= np.amin(field)
    return field.astype(dtype)


def _diamondStep(field, temp_field_size, size, half, n, roughness, height):
    """Diamond-square algorithm - diamond step."""

    for i in range(half, temp_field_size-1, size):
        for j in range(half, temp_field_size-1, size):
            offset = ((np.random.rand() - 0.5) * roughness * height) / 2**n
            field[i, j] = np.mean([field[i + half, j + half],
                                   field[i + half, j - half],
                                   field[i - half, j + half],
                                   field[i - half, j - half]]) + offset
    return field


def _squareStep(field, temp_field_size, size, half, n, roughness, height):
    """Diamond-square algorithm - square step."""

    for i in range(0, temp_field_size, half):
        for j in range((i+half) % size, temp_field_size, size):
            offset = ((np.random.rand() - 0.5) * roughness * height) / 2**n
            if i == 0:
                field[i, j] = np.mean([field[i, j + half],
                                       field[i, j - half],
                                       field[i + half, j]]) + offset
            elif i == temp_field_size-1:
                field[i, j] = np.mean([field[i, j + half],
                                       field[i, j - half],
                                       field[i - half, j]]) + offset
            elif j == 0:
                field[i, j] = np.mean([field[i, j + half],
                                       field[i + half, j],
                                       field[i - half, j]]) + offset
            elif j == temp_field_size-1:
                field[i, j] = np.mean([field[i, j - half],
                                       field[i + half, j],
                                       field[i - half, j]]) + offset
            else:
                field[i, j] = np.mean([field[i, j + half],
                                       field[i, j - half],
                                       field[i + half, j],
                                       field[i - half, j]]) + offset
    return field


def diamondSquare(field_shape, height, roughness, dtype=None):
    """Generate random field using diamond-square algorithm."""

    if field_shape[0] != field_shape[1]:
        raise ValueError("Random terrain only compatible with square fields.")

    next_pow_2 = lambda x: math.ceil(math.log2(abs(x)))

    # Padded field size determined by next 2^n + 1
    temp_field_size = 2**next_pow_2(field_shape[0] - 1) + 1

    # Maximum possible iterations
    iteration_count = next_pow_2(temp_field_size - 1)

    # Initialize field
    field = np.zeros((temp_field_size, temp_field_size))

    # Seed corner values
    field[ 0,  0] = np.random.rand() * roughness * height
    field[-1,  0] = np.random.rand() * roughness * height
    field[ 0, -1] = np.random.rand() * roughness * height
    field[-1, -1] = np.random.rand() * roughness * height

    # Perform diamond-square algorithm
    size = temp_field_size - 1
    for n in range(iteration_count):
        half = int(size / 2)
        field = _diamondStep(field, temp_field_size, size, half, n, roughness, height)
        field =  _squareStep(field, temp_field_size, size, half, n, roughness, height)
        size = half

    # Remove padding
    padding = temp_field_size - field_shape[0] - 1
    field = field[padding:temp_field_size-1, padding:temp_field_size-1]

    # Shift field vertically by setting lowest point to 0
    field -= np.amin(field)

    return field.astype(dtype)


def regularPolygon(sides, radius, rotation=0, translation=(0, 0)):
    """Generate regular shapely polygon."""

    theta = 2 * np.pi / sides
    points = [[np.sin(theta * i + rotation) * radius + translation[0],
               np.cos(theta * i + rotation) * radius + translation[1]] for i in range(sides)]
    
    return geometry.Polygon(points)


def rectangle(x_center, y_center, width, height):
    """Generate shapely rectangle."""
    points = [
        [x_center - width/2, y_center - height/2],
        [x_center - width/2, y_center + height/2],
        [x_center + width/2, y_center + height/2],
        [x_center + width/2, y_center - height/2]]
    return geometry.Polygon(points)


def setBorder(field, thickness, value):
    """Set border of field with given thickness to given value."""

    if thickness < 0:
        raise ValueError("Border thickness must be positive")
    elif thickness == 0:
        return field.copy
    field_new = field.copy()
    field_new[0:thickness, :] = value
    field_new[-thickness:, :] = value
    field_new[:, 0:thickness] = value
    field_new[:, -thickness:] = value
    return field_new


def setLine(field, x0, y0, x1, y1, width, value):
    """Set line through field with given width to given value."""

    x_vec = np.arange(field.shape[0])
    y_vec = np.arange(field.shape[1])

    x = np.array([x1-x0, y1-y0])

    field_new = field.copy()
    for i, x in enumerate(x_vec):
        for j, y in enumerate(y_vec):
            dx = np.array([x-x0, y-y0]) # Vector from point 0 to query point
            dx1 = np.dot(dx, x) / np.dot(x, x) * x # Projection of dx onto x
            dx2 = x - dx1 # Rejection of dx onto x
            dx3 = np.array([x-x1, y-y1]) # Vector from point 1 to query point
            in_width = np.linalg.norm(dx1) <= width/2
            in_length = (np.dot(dx2, x) >= 0) and (np.linalg.norm(dx2) <= np.linalg.norm(x))
            in_cap = (np.linalg.norm(dx) <= width/2) or (np.linalg.norm(dx3) <= width/2)
            if (in_width and in_length) or in_cap:
                field_new[i,j] = value
    return field_new