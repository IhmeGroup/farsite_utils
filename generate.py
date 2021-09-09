"""Generate random 2D fields."""

import math
import numpy as np
from shapely.geometry import Polygon


def randomUniform(field_shape, low, high):
    """Generate field with uniformly-distrubuted noise."""

    return np.random.uniform(low=low, high=high, size=field_shape)


def randomInteger(field_shape, low, high):
    """Generate field with uniformly-distributed integer noise."""

    return np.random.randint(low=low, high=high, size=field_shape, dtype=np.int32)


def randomNormal(field_shape, mean=0, stddev=0.25):
    """Generate field with normally-distrubuted noise."""

    return np.random.normal(loc=mean, scale=stddev, size=field_shape)


def randomFoldedNormal(field_shape, mean=1, stddev=0.25):
    """Generate field with folded normally-distributed noise."""

    return np.abs(randomNormal(field_shape, mean, stddev))


def randomBool(field_shape, p=0.5, out_type=bool):
    """Generate field with boolean values, given probability of True."""

    raw = np.random.rand(*field_shape)
    field = np.less_equal(raw, p)
    return field.astype(out_type)


def randomChoice(field_shape, vals):
    """Generate field selecting random value from list of given vals."""

    return np.random.choice(vals, field_shape)


def randomPatchy(field_shape, d, p_large, p_small, mean=1, stdev=0.25):
    """Generate patchy field."""

    A_field = field_shape[0] * field_shape[1]
    A_patch = np.pi * d**2

    field = np.zeros(field_shape)
    field_with_patches = np.zeros(field_shape, dtype=bool)
    p_actual = 0

    x = np.arange(field_shape[0])
    y = np.arange(field_shape[1])
    X,Y = np.meshgrid(x,y)
    
    while p_actual <= p_large:
        loc = np.round(np.random.rand(2) * np.array(field_shape))

        X_dist = X - loc[0]
        Y_dist = Y - loc[1]
        R = np.sqrt(X_dist**2 + Y_dist**2)
        
        patch_bool = np.less_equal(R, d/2)
        field_with_patches = np.logical_or(field_with_patches, patch_bool)

        p_actual = np.sum(field_with_patches) / (field_shape[0] * field_shape[1])

    field = field_with_patches * np.random.normal(loc=mean, scale=stdev, size=field_shape)
    field *= density_bool(field_shape, p_small)
    field = np.abs(field)
    return field


def gradient(field_shape, aspect, slope):
    """Generate field with constant given slope in given direction.
    All angles in radians."""

    plane_norm = np.array([-np.sin(slope) * np.cos(-aspect + np.pi/2),
                           -np.sin(slope) * np.sin(-aspect + np.pi/2),
                           np.cos(slope)])
    
    x = np.arange(field_shape[0])
    y = np.arange(field_shape[1])
    [X, Y] = np.meshgrid(x, y)

    field = (plane_norm[0]*X + plane_norm[1]*Y) / plane_norm[2]
    return field - np.amin(field)


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


def diamondSquare(field_shape, height, roughness):
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
    return field - np.amin(field)


def regularPolygon(sides, radius, rotation=0, translation=(0, 0)):
    theta = 2 * np.pi / sides
    points = [[np.sin(theta * i + rotation) * radius + translation[0],
               np.cos(theta * i + rotation) * radius + translation[1]] for i in range(sides)]
    
    return Polygon(points)