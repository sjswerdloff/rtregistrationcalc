# ConvertMatrixToEuler
# copied from various code examples at https://learnopencv.com/rotation-matrix-to-euler-angles
# modified by Stuart Swerdloff to increase the allowable difference of the diagonal from identity
# and added main with hardcoded values as a concrete exercise
# and refactored to make pylint happy

"""Convert a 3x3 matrix to Euler Angle representation

Returns:
    np.ndarray: the Euler angles
"""
import math

import numpy as np


# Checks if a matrix is a valid rotation matrix.
def is_rotation_matrix(rotation_matrix: np.ndarray) -> bool:
    """Checks if a matrix is a valid rotation matrix.
    By transposing, multiplying, and checking the diagonal norm
    Args:
        rotation_matrix (np.ndarray): the rotation matrix

    Returns:
        bool: true if the matrix is close enough to a rotation matrix to be decomposable
    """
    transpose = np.transpose(rotation_matrix)
    should_be_identity = np.dot(transpose, rotation_matrix)
    identity_matrix = np.identity(3, dtype=rotation_matrix.dtype)
    norm = np.linalg.norm(identity_matrix - should_be_identity)
    print(f"difference from identity = {norm}")
    if norm < 2e-6:
        return True
    return False


def rotation_matrix_to_euler_angles(rotation_matrix: np.ndarray) -> np.ndarray:
    """Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).

    Args:
        rotation_matrix (np.ndarray): the 3x3 rotation matrix

    Returns:
        np.ndarray: the euler angles in order Roll, Pitch, Yaw
    """

    assert is_rotation_matrix(rotation_matrix)

    _sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])

    singular = _sy < 1e-6

    if not singular:
        _x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        _y = math.atan2(-rotation_matrix[2, 0], _sy)
        _z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        _x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        _y = math.atan2(-rotation_matrix[2, 0], _sy)
        _z = 0

    return np.array([_x, _y, _z])


# Calculates Rotation Matrix given euler angles.
def euler_angles_to_rotation_matrix(theta: np.ndarray) -> np.ndarray:
    """Calculates Rotation Matrix given euler angles.

    Args:
        theta (np.ndarray): euler angles in order of Roll,Pitch, Yaw (Tait-Bryan)

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    rotation_matrix = np.identity(3, dtype=theta.dtype)
    r_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    r_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    r_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    rotation_matrix = np.dot(r_z, np.dot(r_y, r_x))
    if np.shape(rotation_matrix) != (3, 3):
        raise ValueError("Failure to compute 3x3 matrix")

    return rotation_matrix


if __name__ == "__main__":
    # matrix --> euler angles --> matrix
    sample_rotation_matrix = np.array(
        [
            [0.998984, 0.032327, 0.031397],
            [-0.031397, 0.999067, -0.029666],
            [-0.032327, 0.02865, 0.999067],
        ]
    )
    euler_angles = rotation_matrix_to_euler_angles(sample_rotation_matrix)
    r_prime = euler_angles_to_rotation_matrix(euler_angles)
    in_degrees = euler_angles * 180.0 / math.pi
    print(in_degrees)
    print(r_prime)

    print("")
    # degree angles to matrix
    hitachi_theta_degrees = np.array([4.2, 4.6, 0.1])
    print(
        f"Hitachi Pitch {hitachi_theta_degrees[1]} \
  Hitachi Roll {hitachi_theta_degrees[0]} \
  Hitachi Yaw {hitachi_theta_degrees[2]}"
    )
    hitachi_theta = hitachi_theta_degrees * math.pi / 180.0
    print("Hitachi Rotation Matrix")
    hitachi_r = euler_angles_to_rotation_matrix(hitachi_theta)
    print(hitachi_r)
