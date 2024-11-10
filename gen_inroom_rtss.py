#!/usr/bin/env python

import glob
import logging
import sys
from datetime import datetime
from os import path as os_path
from pathlib import Path
from typing import Dict, List

import numpy as np
from pydicom import Dataset, Sequence, read_file, uid, write_file

#  Copied and modified from ImageLoading.py from OnkoDICOM, which was LGPL 2.1 at the time


def img_stack_displacement(orientation: List[str], position: List[str]) -> float:
    """
    Calculate the projection of the image position patient along the
    axis perpendicular to the images themselves, i.e. along the stack
    axis. Intended use is for the sorting key to sort a stack of image
    datasets so that they are in order, regardless of whether the images
    are axial, coronal, or sagittal, and independent from the order in
    which the images were read in.

    :param orientation: List of strings with six elements, the image
        orientation patient value from the dataset.
    :param position: List of strings with three elements, the image
    position value from the dataset.
    :return: Float of the image position patient along the image stack
        axis.
    """
    ds_orient_x = orientation[0:3]
    ds_orient_y = orientation[3:6]
    orient_x = np.array(list(map(float, ds_orient_x)))
    orient_y = np.array(list(map(float, ds_orient_y)))
    orient_z = np.cross(orient_x, orient_y)
    img_pos_patient = np.array(list(map(float, position)))
    displacement = orient_z.dot(img_pos_patient)

    return displacement


def get_dict_sort_on_displacement(item: tuple[str, Dataset]) -> float:
    """

    :param item: dictionary key, value item with value of a PyDicom
        dataset
    :return: Float of the projection of the image position patient on
        the axis through the image stack
    """
    img_dataset = item[1]
    orientation = img_dataset.ImageOrientationPatient
    position = img_dataset.ImagePositionPatient
    sort_key = img_stack_displacement(orientation, position)

    return sort_key


def image_stack_sort(read_data_dict: Dict[str, Dataset]) -> List[tuple[str, Dataset]]:
    """
    Sort the read_data_dict by order of displacement
    along the image stack axis. For axial images this is by the Z
    coordinate.
    :return: Tuple of sorted dictionaries
    """
    new_items = read_data_dict.items()
    sorted_dict_on_displacement = sorted(new_items, key=get_dict_sort_on_displacement, reverse=True)
    return sorted_dict_on_displacement


def get_stack_center(sorted_dict_on_displacement: List[tuple[str, Dataset]]) -> List[float]:
    first_ds = sorted_dict_on_displacement[0][1]
    last_ds = sorted_dict_on_displacement[len(sorted_dict_on_displacement) - 1][1]
    first_image_pos = np.array(list(map(float, first_ds.ImagePositionPatient)))
    logging.debug(f"First image position: {first_image_pos}")
    last_image_pos = np.array(list(map(float, last_ds.ImagePositionPatient)))
    logging.debug(f"Last Image Position: {last_image_pos}")
    row_spacing = last_ds.PixelSpacing[0]
    logging.debug(f"Row spacing: {row_spacing}")
    column_spacing = last_ds.PixelSpacing[1]  # see https://dicom.innolitics.com/ciods/ct-image/image-plane/00280030
    logging.debug(f"Column spacing: {column_spacing}")
    rows = last_ds.Rows
    logging.debug(f"Number of Rows: {rows}")
    cols = last_ds.Columns
    logging.debug(f"Number of Columns: {cols}")
    # image_width = last_ds.Columns * column_spacing
    # image_height = last_ds.Rows * row_spacing

    logging.debug(f"Patient Position (with respect to gravity and the Gantry): {last_ds.PatientPosition}")
    orientation = last_ds.ImageOrientationPatient
    logging.debug(f"Image Orientation Patient: {orientation}")
    ds_orient_x = orientation[0:3]
    ds_orient_y = orientation[3:6]
    orient_x = np.array(list(map(float, ds_orient_x)))
    orient_y = np.array(list(map(float, ds_orient_y)))

    column_last_pixel_displacement = (cols - 1) * orient_x * column_spacing
    row_last_pixel_displacement = (rows - 1) * orient_y * row_spacing
    last_image_last_pixel_pos = last_image_pos + column_last_pixel_displacement + row_last_pixel_displacement
    image_stack_isocenter_pos = 0.5 * (last_image_last_pixel_pos + first_image_pos)
    return image_stack_isocenter_pos.tolist()


def load_ct_headers_from_directory(ct_directory: Path) -> Dict[Path, Dataset]:
    files = list_files(ct_directory, "dcm")
    ds_dict = {}
    for file in files:
        ds = read_file(file, force=True, stop_before_pixels=True)
        if ds.SOPClassUID == uid.CTImageStorage:
            ds_dict[file] = ds
    return ds_dict


def list_files(filepath: Path, filetype: str) -> List[Path]:
    paths = []
    str_glob = f"*.{filetype}"
    for name in glob.glob(os_path.join(str(filepath), str_glob)):
        paths.append(name)
    return paths


def get_stack_center_from_path(ct_directory: Path) -> List[float]:
    dict_of_ct_headers = load_ct_headers_from_directory(ct_directory)
    sorted_stack = image_stack_sort(dict_of_ct_headers)
    ct_stack_center = get_stack_center(sorted_stack)
    logging.debug(f"CT volume with {len(sorted_stack)} slices in {ct_directory} is centered at {ct_stack_center}")
    return ct_stack_center


def usage():
    print(f"{sys.argv[0]} ct_directory rt_ion_plan_file_path ref_rtss_file_path")
    print("The ct_directory is used to find the CBCT isocenter and to provide patient and study information")
    print("The RT Ion Plan is used to identify the (original) Frame of Reference UID and to validate the referenced RT SS UID")
    print("The Referenced RT Structure Set is used to identify the Series and SOP Instance UIDs of the reference CT")


def pre_populate_inroom_rtss_header(ct_ds: Dataset, inroom_rtss_ds: Dataset = None) -> Dataset:
    if inroom_rtss_ds is None:
        prepopulated_rtss_ds = Dataset()
    else:
        prepopulated_rtss_ds = inroom_rtss_ds

    for key_word in [
        "StudyDate",
        "StudyTime",
        "AccessionNumber",
        "Manufacturer",
        "InstitutionName",
        "InstitutionAddress",
        "ReferringPhysiciansName",
        "OperatorsName",
        "PatientID",
        "PatientsBirthDate",
        "PatientsSex",
        "StudyInstanceUID",
        "StudyID",
    ]:
        try:
            prepopulated_rtss_ds[key_word] = ct_ds[key_word]
        except (KeyError, IndexError):
            print(f"{key_word} not found in CT")

    return prepopulated_rtss_ds


def populate_ifsseq0099_rtss(sorted_stack, ct_stack_center, inroom_rtss_ds):
    now = datetime.now()
    first_ct_ds = sorted_stack[0][1]
    inroom_rtss_ds.StructureSetROISequence = Sequence()
    inroom_rtss_ds.ROIContourSequence = Sequence()
    inroom_rtss_ds.RTROIObservationsSequence = Sequence()
    inroom_rtss_ds.ReferencedFrameOfReferenceSequence = Sequence()

    inroom_rtss_ds.InstanceCreationDate = now.strftime("%Y%m%d")
    inroom_rtss_ds.InstanceCreationTime = now.strftime("%H%M%S")
    inroom_rtss_ds.SOPClassUID = uid.RTStructureSetStorage
    inroom_rtss_ds.SOPInstanceUID = uid.generate_uid()  # could potentially use org root of CT as prefix
    inroom_rtss_ds.SeriesInstanceUID = uid.generate_uid()
    inroom_rtss_ds.Modality = "RTSTRUCT"
    inroom_rtss_ds.StructureSetLabel = "InRoom Isocenter"
    inroom_rtss_ds.StructureSetName = "RTSS for Setup CBCT"
    inroom_rtss_ds.StructureSetDescription = "IFSSEQ0099 compliant RT SS for positioning CT"
    inroom_rtss_ds.StructureSetDate = now.strftime("%Y%m%d")
    inroom_rtss_ds.StructureSetTime = now.strftime("%H%M%S")

    ref_frame_reference_sequence_item = Dataset()
    ref_frame_reference_sequence_item.FrameOfReferenceUID = first_ct_ds.FrameOfReferenceUID
    ref_frame_reference_sequence_item.RTReferencedStudySequence = Sequence()
    ref_study_sequence_item = Dataset()
    ref_study_sequence_item.ReferencedSOPClassUID = first_ct_ds.SOPClassUID
    ref_study_sequence_item.ReferencedSOPInstanceUID = first_ct_ds.StudyInstanceUID
    ref_study_sequence_item.ReferencedSeriesSequence = Sequence()
    ref_series_sequence_item = Dataset()
    ref_series_sequence_item.SeriesInstanceUID = first_ct_ds.SeriesInstanceUID
    ref_series_sequence_item.ContourImageSequence = Sequence()
    for ct_tuple in sorted_stack:
        ct_ds = ct_tuple[1]
        contour_sequence_item = Dataset()
        contour_sequence_item.ReferencedSOPClassUID = ct_ds.SOPClassUID
        contour_sequence_item.ReferencedSOPInstanceUID = ct_ds.SOPInstanceUID
        ref_series_sequence_item.ContourImageSequence.append(contour_sequence_item)

    ref_study_sequence_item.ReferencedSeriesSequence.append(ref_series_sequence_item)
    ref_frame_reference_sequence_item.RTReferencedStudySequence.append(ref_study_sequence_item)
    inroom_rtss_ds.ReferencedFrameOfReferenceSequence.append(ref_frame_reference_sequence_item)

    ss_roi_sequence_item = Dataset()
    ss_roi_sequence_item.ROINumber = 1
    ss_roi_sequence_item.ReferencedFrameOfReferenceUID = first_ct_ds.FrameOfReferenceUID
    ss_roi_sequence_item.ROIName = "InitMatchIso"  # See IFSSEQ0099
    ss_roi_sequence_item.ROIDescription = "Isocenter of Treatment Machine"
    ss_roi_sequence_item.ROIGenerationAlgorithm = "AUTOMATIC"
    ss_roi_sequence_item.ROIGenerationDescription = "Extracted from Center of CBCT Image Volume"

    inroom_rtss_ds.StructureSetROISequence.append(ss_roi_sequence_item)

    ss_roi_sequence_item = Dataset()
    ss_roi_sequence_item.ROINumber = 2
    ss_roi_sequence_item.ReferencedFrameOfReferenceUID = first_ct_ds.FrameOfReferenceUID
    ss_roi_sequence_item.ROIName = "SetupIsocenter"  # See IFSSEQ0099
    ss_roi_sequence_item.ROIDescription = "Isocenter of Treatment Machine"
    ss_roi_sequence_item.ROIGenerationAlgorithm = "AUTOMATIC"
    ss_roi_sequence_item.ROIGenerationDescription = "Extracted from Center of CBCT Image Volume"

    inroom_rtss_ds.StructureSetROISequence.append(ss_roi_sequence_item)

    roi_contour_sequence_item = Dataset()
    roi_contour_sequence_item.ReferencedROINumber = 1
    roi_contour_sequence_item.ContourSequence = Sequence()
    contour_sequence_item = Dataset()
    contour_sequence_item.ContourNumber = 1
    contour_sequence_item.ContourGeometricType = "POINT"
    contour_sequence_item.NumberOfContourPoints = 1
    contour_sequence_item.ContourData = ct_stack_center

    roi_contour_sequence_item.ContourSequence.append(contour_sequence_item)
    inroom_rtss_ds.ROIContourSequence.append(roi_contour_sequence_item)

    roi_contour_sequence_item = Dataset()
    roi_contour_sequence_item.ReferencedROINumber = 2
    roi_contour_sequence_item.ContourSequence = Sequence()
    contour_sequence_item = Dataset()
    contour_sequence_item.ContourNumber = 1
    contour_sequence_item.ContourGeometricType = "POINT"
    contour_sequence_item.NumberOfContourPoints = 1
    contour_sequence_item.ContourData = ct_stack_center
    roi_contour_sequence_item.ContourSequence.append(contour_sequence_item)
    inroom_rtss_ds.ROIContourSequence.append(roi_contour_sequence_item)

    rt_roi_observations_sequence_item = Dataset()
    rt_roi_observations_sequence_item.ObservationNumber = 1
    rt_roi_observations_sequence_item.ReferencedROINumber = 1
    rt_roi_observations_sequence_item.RTROIInterpretedType = "INITMATCHISO"  # See IFSSEQ0099
    rt_roi_observations_sequence_item.ROIInterpreter = ""  # use None instead of ""  ?
    inroom_rtss_ds.RTROIObservationsSequence.append(rt_roi_observations_sequence_item)

    rt_roi_observations_sequence_item = Dataset()
    rt_roi_observations_sequence_item.ObservationNumber = 2
    rt_roi_observations_sequence_item.ReferencedROINumber = 2
    rt_roi_observations_sequence_item.RTROIInterpretedType = "SETUPISOCENTER"  # See IFSSEQ0099
    rt_roi_observations_sequence_item.ROIInterpreter = ""  # use None instead of ""  ?
    inroom_rtss_ds.RTROIObservationsSequence.append(rt_roi_observations_sequence_item)


if __name__ == "__main__":
    num_args = len(sys.argv)
    if num_args < 2:
        usage()
        sys.exit("No arguments provided, must at least provide directory where in-room CT/CBCT data files are.")
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s|%(name)s|%(levelname)s|%(funcName)s|%(message)s")
    ct_directory = Path(sys.argv[1]).expanduser()
    if not ct_directory.exists():
        sys.exit(f"Unable to find {ct_directory}")
    # ct_stack_center = get_stack_center_from_path(ct_directory)
    dict_of_ct_headers = load_ct_headers_from_directory(ct_directory)
    sorted_stack = image_stack_sort(dict_of_ct_headers)
    ct_stack_center = get_stack_center(sorted_stack)
    print(ct_stack_center)
    if num_args < 4:
        usage()
        sys.exit()
    else:
        ion_plan_ds = read_file(Path(sys.argv[2]).expanduser(), force=True)
        ref_rt_ss = read_file(Path(sys.argv[3]).expanduser(), force=True)
        plan_ref_rtss = str(ion_plan_ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID)
        ref_rtss_uid = str(ref_rt_ss.SOPInstanceUID)
        if plan_ref_rtss != ref_rtss_uid:
            sys.exit(f"Referenced RT SS in plan: {plan_ref_rtss} doesn't match RT SS UID: {ref_rtss_uid}")

    now = datetime.now()
    # Pre-populate the inroom RT SS with data from the CT
    # Patient and Study Information
    first_ct_ds = sorted_stack[0][1]
    inroom_rtss_ds = pre_populate_inroom_rtss_header(first_ct_ds)

    populate_ifsseq0099_rtss(sorted_stack, ct_stack_center, inroom_rtss_ds)

    inroom_rtss_ds.is_implicit_VR = True
    inroom_rtss_ds.is_little_endian = True
    write_file(f"RS_{inroom_rtss_ds.SOPInstanceUID}.dcm", inroom_rtss_ds)
