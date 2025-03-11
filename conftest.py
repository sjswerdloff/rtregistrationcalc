import pytest
import os
import tempfile
import numpy as np
from pathlib import Path
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

@pytest.fixture
def create_temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def create_mock_ct_dataset():
    """Create a mock CT DICOM dataset."""
    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    ds.StudyInstanceUID = "1.2.3.4.5.6.7.8.9"
    ds.SeriesInstanceUID = "1.2.3.4.5.6.7.8.9.1"
    ds.FrameOfReferenceUID = "1.2.3.4.5.6.7.8.9.2"

    ds.PatientID = "TEST123"
    ds.PatientName = "TEST^PATIENT"
    ds.PatientsBirthDate = "19800101"
    ds.PatientsSex = "O"

    ds.StudyDate = "20230101"
    ds.StudyTime = "120000"
    ds.AccessionNumber = "ACC12345"
    ds.StudyID = "STUDY01"

    ds.Manufacturer = "TESTMANUF"
    ds.InstitutionName = "TEST HOSPITAL"

    # Image specific attributes
    ds.ImageOrientationPatient = ["1.0", "0.0", "0.0", "0.0", "1.0", "0.0"]
    ds.ImagePositionPatient = ["0.0", "0.0", "0.0"]
    ds.PixelSpacing = [1.0, 1.0]
    ds.Rows = 512
    ds.Columns = 512
    ds.PatientPosition = "HFS"  # Head First Supine

    return ds


def create_matrix_item():
    """Create the matrix item with the 4x4 transformation matrix."""
    matrix_item = Dataset()
    matrix_item.FrameOfReferenceTransformationMatrix = [
        0.999, 0.012, 0.008, 10.0,   # First row
        -0.010, 0.998, 0.015, -5.0,   # Second row
        -0.009, -0.014, 0.999, 2.5,   # Third row
        0.0, 0.0, 0.0, 1.0            # Fourth row
    ]
    return matrix_item


def build_registration_sequence(matrix_item):
    """Compose the nested registration sequence using the matrix item."""
    matrix_seq = Sequence([matrix_item])
    matrix_reg_item = Dataset()
    matrix_reg_item.MatrixSequence = matrix_seq
    matrix_reg_seq = Sequence([matrix_reg_item])
    reg_item = Dataset()
    reg_item.MatrixRegistrationSequence = matrix_reg_seq
    return Sequence([reg_item])


@pytest.fixture
def create_mock_registration_dataset():
    """Create a mock registration dataset with transformation matrix."""
    reg_ds = Dataset()
    matrix_item = create_matrix_item()
    reg_ds.RegistrationSequence = build_registration_sequence(matrix_item)
    return reg_ds
