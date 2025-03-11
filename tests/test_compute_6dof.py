import pytest
import numpy as np
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from compute_6dof_from_reg_rtss_plan import (
    compute_6dof_from_reg_rtss_plan,
    convert_dicom_patient_ypr_to_iec_ypr,
    convert_dicom_patient_to_iec
)


class TestCompute6DOF:
    @pytest.fixture
    def mock_reg_ds(self):
        """Create a mock registration dataset with transformation matrix."""
        reg_ds = Dataset()

        # Create nested sequences
        reg_seq = Sequence()
        matrix_reg_seq = Sequence()
        matrix_seq = Sequence()

        # Create the matrix item with transformation matrix
        matrix_item = Dataset()
        # Sample 4x4 transformation matrix in DICOM flat format
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
        reg_ds.RegistrationSequence = reg_seq

        return reg_ds

    @pytest.fixture
    def mock_rtss_ds(self):
        """Create a mock RTSS dataset with setup isocenter."""
        rtss_ds = Dataset()

        # Create ROI Contour Sequence
        roi_contour_seq = Sequence()
        contour_seq = Sequence()
        contour_item = Dataset()
        contour_item.ContourGeometricType = "POINT"
        contour_item.ContourData = ["100.0", "200.0", "300.0"]
        contour_item.NumberOfContourPoints = 1
        contour_seq.append(contour_item)

        roi_contour_item = Dataset()
        roi_contour_item.ContourSequence = contour_seq
        roi_contour_item.ReferencedROINumber = 2
        roi_contour_seq.append(roi_contour_item)

        # Create Structure Set ROI Sequence
        struct_set_roi_seq = Sequence()
        roi_item = Dataset()
        roi_item.ROINumber = 2
        roi_item.ROIName = "SetupIsocenter"
        struct_set_roi_seq.append(roi_item)

        # Create RT ROI Observations Sequence
        rt_roi_observations_seq = Sequence()
        obs_item = Dataset()
        obs_item.ReferencedROINumber = 2
        obs_item.RTROIInterpretedType = "SETUPISOCENTER"
        rt_roi_observations_seq.append(obs_item)

        rtss_ds.ROIContourSequence = roi_contour_seq
        rtss_ds.StructureSetROISequence = struct_set_roi_seq
        rtss_ds.RTROIObservationsSequence = rt_roi_observations_seq

        return rtss_ds

    @pytest.fixture
    def mock_plan_ds(self):
        """Create a mock RT Plan dataset with beam isocenter."""
        plan_ds = Dataset()

        # Create Ion Beam Sequence
        beam_seq = Sequence()
        beam_item = Dataset()
        beam_item.IsocenterPosition = ["105.0", "195.0", "305.0"]
        
        # Add IonControlPointSequence for the beam
        ion_control_point_seq = Sequence()
        control_point_item = Dataset()
        control_point_item.IsocenterPosition = ["105.0", "195.0", "305.0"]
        control_point_item.PatientSupportAngle = 0.0
        ion_control_point_seq.append(control_point_item)
        beam_item.IonControlPointSequence = ion_control_point_seq
        
        beam_seq.append(beam_item)
        plan_ds.IonBeamSequence = beam_seq
        
        # Create PatientSetupSequence
        patient_setup_seq = Sequence()
        patient_setup_item = Dataset()
        patient_setup_item.PatientPosition = "HFS"  # Head First Supine
        patient_setup_seq.append(patient_setup_item)
        plan_ds.PatientSetupSequence = patient_setup_seq

        return plan_ds

    def test_convert_dicom_patient_ypr_to_iec_ypr(self):
        """Test conversion of DICOM Patient YPR to IEC Table YPR."""
        # Sample Euler angles in DICOM Patient coordinates
        dicom_ypr = np.array([1.0, 2.0, 3.0])
        
        # Test with HFS (Head First Supine) patient position
        iec_ypr_hfs = convert_dicom_patient_ypr_to_iec_ypr(dicom_ypr, "HFS")
        assert iec_ypr_hfs[0] == dicom_ypr[0]  # Yaw preserved in HFS
        assert iec_ypr_hfs[1] == dicom_ypr[1]  # Pitch preserved in HFS
        assert iec_ypr_hfs[2] == dicom_ypr[2]  # Roll preserved in HFS
        
        # Test with HFP (Head First Prone) patient position
        iec_ypr_hfp = convert_dicom_patient_ypr_to_iec_ypr(dicom_ypr, "HFP")
        assert iec_ypr_hfp[0] == -dicom_ypr[0]  # Yaw sign flipped in HFP
        assert iec_ypr_hfp[1] == -dicom_ypr[1]  # Pitch sign flipped in HFP
        assert iec_ypr_hfp[2] == dicom_ypr[2]   # Roll preserved in HFP

    def test_convert_dicom_patient_to_iec(self):
        """Test conversion of DICOM Patient translation to IEC coordinates."""
        # Sample translation in DICOM Patient coordinates
        dicom_translation = np.array([10.0, 20.0, 30.0])

        # Test with HFS (Head First Supine) patient position
        iec_translation_hfs = convert_dicom_patient_to_iec(dicom_translation, "HFS")
        
        # Check HFS conversion according to the function
        assert iec_translation_hfs[0] == dicom_translation[0]     # X = X(DICOM)
        assert iec_translation_hfs[1] == dicom_translation[2]     # Y = Z(DICOM)
        assert iec_translation_hfs[2] == -dicom_translation[1]    # Z = -Y(DICOM)
        
        # Test with HFP (Head First Prone) patient position
        iec_translation_hfp = convert_dicom_patient_to_iec(dicom_translation, "HFP")
        
        # Check HFP conversion according to the function
        assert iec_translation_hfp[0] == -dicom_translation[0]    # X = -X(DICOM)
        assert iec_translation_hfp[1] == dicom_translation[2]     # Y = Z(DICOM)
        assert iec_translation_hfp[2] == dicom_translation[1]     # Z = Y(DICOM)

    def test_compute_6dof_from_reg_rtss_plan(self, mock_reg_ds, mock_rtss_ds, mock_plan_ds):
        """Test computation of 6DOF transformation from registration, RTSS, and plan."""
        # Compute 6DOF transformation
        rot, trans = compute_6dof_from_reg_rtss_plan(mock_reg_ds, mock_rtss_ds, mock_plan_ds)

        # Verify output types and shapes
        assert isinstance(rot, np.ndarray)
        assert isinstance(trans, np.ndarray)
        assert rot.shape == (3,)
        assert trans.shape == (3,)