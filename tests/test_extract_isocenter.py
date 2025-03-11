import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from extract_rtss_setup_isocenter import extract_rtss_setup_isocenter
from extract_plan_setupbeam_isocenter import extract_plan_setupbeam_isocenter


class TestExtractIsocenter:
    @pytest.fixture
    def mock_rtss_with_isocenter(self):
        """Create a mock RT Structure Set with setup isocenter."""
        rtss = Dataset()
        
        # Create ROI Contour Sequence
        roi_contour_seq = Sequence()
        
        # Create contour item for setupisocenter
        contour_seq = Sequence()
        contour_item = Dataset()
        contour_item.ContourGeometricType = "POINT"
        contour_item.ContourData = ["10.0", "20.0", "30.0"]
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
        
        rtss.ROIContourSequence = roi_contour_seq
        rtss.StructureSetROISequence = struct_set_roi_seq
        rtss.RTROIObservationsSequence = rt_roi_observations_seq
        
        return rtss
    
    @pytest.fixture
    def mock_plan_with_isocenter(self):
        """Create a mock RT Plan with setup beam isocenter."""
        plan = Dataset()
        
        # Create Ion Beam Sequence
        beam_seq = Sequence()
        beam_item = Dataset()
        
        # Create Control Point Sequence
        control_point_seq = Sequence()
        control_point_item = Dataset()
        control_point_item.IsocenterPosition = ["5.0", "15.0", "25.0"]
        control_point_seq.append(control_point_item)
        
        # Attach Control Point Sequence to beam item
        beam_item.IonControlPointSequence = control_point_seq
        
        beam_seq.append(beam_item)
        plan.IonBeamSequence = beam_seq
        
        return plan
    
    def test_extract_rtss_setup_isocenter(self, mock_rtss_with_isocenter):
        """Test extraction of setup isocenter from RT Structure Set."""
        isocenter = extract_rtss_setup_isocenter(mock_rtss_with_isocenter)
        
        assert isocenter is not None
        assert len(isocenter) == 3
        assert isocenter[0] == "10.0"
        assert isocenter[1] == "20.0"
        assert isocenter[2] == "30.0"
    
    def test_extract_rtss_setup_isocenter_missing(self):
        """Test behavior when no setup isocenter is found in RTSS."""
        rtss = Dataset()
        
        # Create empty sequences
        rtss.ROIContourSequence = Sequence()
        rtss.StructureSetROISequence = Sequence()
        rtss.RTROIObservationsSequence = Sequence()
        
        with pytest.raises(ValueError, match="No ROIName '\\['SetupIsocenter', 'InitMatchIso', 'InitLaserIso'\\]' found in StructureSetROISequence"):
            extract_rtss_setup_isocenter(rtss)
    
    def test_extract_plan_setupbeam_isocenter(self, mock_plan_with_isocenter):
        """Test extraction of isocenter from first beam in RT Plan."""
        isocenter = extract_plan_setupbeam_isocenter(mock_plan_with_isocenter)
        
        assert isocenter is not None
        assert len(isocenter) == 3
        assert isocenter[0] == "5.0"
        assert isocenter[1] == "15.0"
        assert isocenter[2] == "25.0"
    
    def test_extract_plan_setupbeam_isocenter_missing(self):
        """Test behavior when no beam sequence is found in plan."""
        plan = Dataset()
        
        # Create empty plan without beam sequence
        with pytest.raises(AttributeError, match="'Dataset' object has no attribute 'IonBeamSequence'"):
            extract_plan_setupbeam_isocenter(plan)