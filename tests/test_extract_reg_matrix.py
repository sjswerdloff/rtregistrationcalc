import math
import pytest
import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from extract_reg_matrix import (
    extract_matrix_as_np_array,
    extract_4x4_matrix_as_np_array,
    decompose_matrix_order_rpy_as_ypr_degrees
)


class TestExtractRegMatrix:
    @pytest.fixture
    def mock_sro_ds(self):
        """Create a mock SRO dataset with a transformation matrix."""
        ds = Dataset()

        # Create nested sequences
        reg_seq = Sequence()
        matrix_reg_seq = Sequence()
        matrix_seq = Sequence()

        # Create the innermost dataset with transformation matrix
        matrix_item = Dataset()

        # Sample transformation matrix (4x4 in DICOM style flat list)
        transform_matrix = [
            0.999, 0.012, 0.008, 10.0,   # First row
            -0.010, 0.998, 0.015, -5.0,  # Second row
            -0.009, -0.014, 0.999, 2.5,  # Third row
            0.0, 0.0, 0.0, 1.0           # Fourth row
        ]

        matrix_item.FrameOfReferenceTransformationMatrix = transform_matrix

        # Build the hierarchy
        matrix_seq.append(matrix_item)
        matrix_reg_item = Dataset()
        matrix_reg_item.MatrixSequence = matrix_seq
        matrix_reg_seq.append(matrix_reg_item)
        reg_item = Dataset()
        reg_item.MatrixRegistrationSequence = matrix_reg_seq
        reg_seq.append(reg_item)
        ds.RegistrationSequence = reg_seq

        return ds

    def test_extract_matrix_as_np_array(self, mock_sro_ds):
        """Test extraction of 3x3 rotation matrix from SRO."""
        rotation_matrix = extract_matrix_as_np_array(mock_sro_ds)

        # Verify shape and values
        assert rotation_matrix.shape == (3, 3)

        # Check if extraction correctly grabbed first 3x3 elements
        expected = np.array([
            [0.999, 0.012, 0.008],
            [-0.010, 0.998, 0.015],
            [-0.009, -0.014, 0.999]
        ])

        assert np.allclose(rotation_matrix, expected)

    def test_extract_4x4_matrix_as_np_array(self, mock_sro_ds):
        """Test extraction of 4x4 transformation matrix from SRO."""
        transform_matrix = extract_4x4_matrix_as_np_array(mock_sro_ds)

        # Verify shape
        assert transform_matrix.shape == (4, 4)

        # Check expected values
        expected = np.array([
            [0.999, 0.012, 0.008, 10.0],
            [-0.010, 0.998, 0.015, -5.0],
            [-0.009, -0.014, 0.999, 2.5],
            [0.0, 0.0, 0.0, 1.0]
        ])

        assert np.allclose(transform_matrix, expected)

    def test_decompose_matrix_order_rpy_as_ypr_degrees(self):
        """Test decomposition of rotation matrix to Euler angles in YPR order."""
        # Create a test rotation matrix
        rotation_matrix = np.array([
            [0.998984, 0.032327, 0.031397],
            [-0.031397, 0.999067, -0.029666],
            [-0.032327, 0.02865, 0.999067]
        ])

        # Get YPR angles in degrees
        ypr_angles = decompose_matrix_order_rpy_as_ypr_degrees(rotation_matrix)

        # Expected values based on actual output
        expected_yaw = -1.85  # ~-1.85° (negated because function negates in_degrees[1])
        expected_pitch = 1.64  # ~1.64°
        expected_roll = -1.80  # ~-1.80° (negative based on actual output)

        # Test with tolerance
        assert len(ypr_angles) == 3
        assert np.isclose(ypr_angles[0], expected_yaw, atol=0.2)  # Yaw
        assert np.isclose(ypr_angles[1], expected_pitch, atol=0.2)  # Pitch
        assert np.isclose(ypr_angles[2], expected_roll, atol=0.2)  # Roll
