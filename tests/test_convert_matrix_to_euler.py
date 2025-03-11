import math
import pytest
import numpy as np

from convert_matrix_to_euler import (
    is_rotation_matrix,
    rotation_matrix_to_euler_angles,
    euler_angles_to_rotation_matrix
)


class TestConvertMatrixToEuler:
    def test_is_rotation_matrix_with_valid_matrix(self):
        # Create a valid rotation matrix
        matrix = np.array([
            [0.998984, 0.032327, 0.031397],
            [-0.031397, 0.999067, -0.029666],
            [-0.032327, 0.02865, 0.999067],
        ])

        # Test valid rotation matrix
        assert is_rotation_matrix(matrix, tolerance_ortho_normality=0.006) is True

    def test_is_rotation_matrix_with_invalid_matrix(self):
        # Create an invalid matrix (not orthogonal)
        matrix = np.array([
            [1.5, 0.5, 0.5],
            [0.5, 1.5, 0.5],
            [0.5, 0.5, 1.5],
        ])

        # Test invalid rotation matrix
        assert is_rotation_matrix(matrix) is False

    def test_rotation_matrix_to_euler_angles(self):
        # Create a known rotation matrix
        matrix = np.array([
            [0.998984, 0.032327, 0.031397],
            [-0.031397, 0.999067, -0.029666],
            [-0.032327, 0.02865, 0.999067],
        ])

        # Calculate Euler angles
        euler_angles = rotation_matrix_to_euler_angles(matrix)

        # Expected values (approximate)
        expected_roll = math.atan2(matrix[2, 1], matrix[2, 2])
        expected_pitch = math.atan2(-matrix[2, 0], math.sqrt(matrix[2, 1]**2 + matrix[2, 2]**2))
        expected_yaw = math.atan2(matrix[1, 0], matrix[0, 0])

        # Test with tolerance
        assert np.isclose(euler_angles[0], expected_roll)
        assert np.isclose(euler_angles[1], expected_pitch)
        assert np.isclose(euler_angles[2], expected_yaw)

    def test_euler_angles_to_rotation_matrix(self):
        # Create example Euler angles (in radians)
        euler_angles = np.array([0.073, 0.080, 0.002])  # ~4.2°, 4.6°, 0.1°

        # Convert to rotation matrix
        matrix = euler_angles_to_rotation_matrix(euler_angles)

        # Convert back to Euler angles
        euler_angles_recovered = rotation_matrix_to_euler_angles(matrix)

        # Test round-trip conversion
        assert np.allclose(euler_angles, euler_angles_recovered, atol=1e-4)

    def test_round_trip_conversion(self):
        # Test several random rotation matrices
        for _ in range(5):
            # Create random Euler angles
            random_angles = np.random.uniform(-math.pi/4, math.pi/4, 3)

            # Convert to rotation matrix
            matrix = euler_angles_to_rotation_matrix(random_angles)

            # Check that result is a valid rotation matrix
            assert is_rotation_matrix(matrix, tolerance_ortho_normality=0.006)

            # Convert back to Euler angles
            recovered_angles = rotation_matrix_to_euler_angles(matrix)

            # Check that angles are recovered
            assert np.allclose(random_angles, recovered_angles, atol=1e-4)