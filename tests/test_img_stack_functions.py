import pytest
import numpy as np
import os
from pathlib import Path
import tempfile
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from gen_inroom_rtss import (
    img_stack_displacement,
    get_dict_sort_on_displacement,
    image_stack_sort,
    get_stack_center
)

AXIAL_ORIENTATION = ["1.0", "0.0", "0.0", "0.0", "1.0", "0.0"]
CORONAL_ORIENTATION = ["1.0", "0.0", "0.0", "0.0", "0.0", "1.0"]
SAGITTAL_ORIENTATION = ["0.0", "1.0", "0.0", "0.0", "0.0", "1.0"]

class TestImgStackFunctions:
    # Test img_stack_displacement with different orientation/position combinations
    def test_img_stack_displacement_axial(self):
        """Test displacement calculation for axial orientation."""
        # Typical axial orientation
        orientation = AXIAL_ORIENTATION
        position = ["0.0", "0.0", "10.0"]
        result = img_stack_displacement(orientation, position)
        assert result == 10.0

    def test_img_stack_displacement_coronal(self):
        """Test displacement calculation for coronal orientation."""
        # Typical coronal orientation
        orientation = CORONAL_ORIENTATION
        position = ["0.0", "5.0", "0.0"]
        result = img_stack_displacement(orientation, position)
        # The cross product causes the sign to be negative
        assert result == -5.0

    def test_img_stack_displacement_sagittal(self):
        """Test displacement calculation for sagittal orientation."""
        # Typical sagittal orientation
        orientation = SAGITTAL_ORIENTATION
        position = ["2.0", "0.0", "0.0"]
        result = img_stack_displacement(orientation, position)
        assert result == 2.0

    # Test dictionary sorting based on displacement
    def test_get_dict_sort_on_displacement(self):
        """Test sorting key function for displacement."""
        # Create a mock dataset
        ds = Dataset()
        ds.ImageOrientationPatient = AXIAL_ORIENTATION
        ds.ImagePositionPatient = ["0.0", "0.0", "5.0"]

        # Create a mock tuple (key, dataset)
        item = ("test_key", ds)

        # Get sorting key
        result = get_dict_sort_on_displacement(item)
        assert result == 5.0

    # Test image stack sorting
    def test_image_stack_sort(self):
        """Test sorting of image stack by displacement."""
        # Create mock datasets with different positions
        ds1 = Dataset()
        ds1.ImageOrientationPatient = AXIAL_ORIENTATION
        ds1.ImagePositionPatient = ["0.0", "0.0", "0.0"]

        ds2 = Dataset()
        ds2.ImageOrientationPatient = AXIAL_ORIENTATION
        ds2.ImagePositionPatient = ["0.0", "0.0", "5.0"]

        ds3 = Dataset()
        ds3.ImageOrientationPatient = AXIAL_ORIENTATION
        ds3.ImagePositionPatient = ["0.0", "0.0", "10.0"]

        # Create dictionary of datasets
        read_data_dict = {
            "file1.dcm": ds1,
            "file2.dcm": ds2,
            "file3.dcm": ds3
        }

        # Sort the dictionary
        sorted_items = image_stack_sort(read_data_dict)

        # Since we're sorting in reverse order (per the implementation),
        # we expect the highest displacement value to be first
        assert sorted_items[0][0] == "file3.dcm"
        assert sorted_items[1][0] == "file2.dcm"
        assert sorted_items[2][0] == "file1.dcm"

    # Test get_stack_center calculation
    def test_get_stack_center(self):
        """Test calculation of stack center."""
        # Create mock datasets
        ds_first = Dataset()
        ds_first.ImageOrientationPatient = AXIAL_ORIENTATION
        ds_first.ImagePositionPatient = ["0.0", "0.0", "0.0"]

        ds_last = Dataset()
        ds_last.ImageOrientationPatient = AXIAL_ORIENTATION
        ds_last.ImagePositionPatient = ["0.0", "0.0", "20.0"]
        ds_last.PixelSpacing = [1.0, 1.0]
        ds_last.Rows = 256
        ds_last.Columns = 256
        ds_last.PatientPosition = "HFS"  # Head First Supine

        # Create sorted dictionary items
        sorted_items = [
            ("first.dcm", ds_first),
            ("last.dcm", ds_last)
        ]

        # Calculate stack center
        center = get_stack_center(sorted_items)

        # Expected center values for a 256x256 image with 1mm spacing
        # The 0.5 * (first_pos + last_pos + extents) calculation
        expected_x = 0.5 * (0.0 + 0.0 + 255)
        expected_y = 0.5 * (0.0 + 0.0 + 255)
        expected_z = 0.5 * (0.0 + 20.0)

        assert np.isclose(center[0], expected_x)
        assert np.isclose(center[1], expected_y)
        assert np.isclose(center[2], expected_z)
