# Copyright (C) 2023 Stuart Swerdloff
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Top level calculation of the 6DOF offset from well known DICOM objects

Raises:
    ValueError: When DICOM Patient Position value in use is not supported

Returns:
    a pair of ndarray: two 3D vectors, rotations and translations
"""

import sys
from typing import Tuple

import numpy as np
import pydicom

import extract_plan_setupbeam_isocenter as ep
import extract_reg_matrix as er
import extract_rtss_setup_isocenter as ertss


def compute_6dof_from_reg_rtss_plan(
    reg_ds: pydicom.Dataset, rtss_ds: pydicom.Dataset, plan_ds: pydicom.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        reg_ds (pydicom.Dataset): dataset representing the Spatial Registration Object
        rtss_ds (pydicom.Dataset): dataset representing the RT Structure Set for the in room image volume
        plan_ds (pydicom.Dataset): dataset representing the RT Ion Plan (containing the planned setup isocenter)

    Returns:
        The correction in IEC61217 Table Top as a pair of np.arrays,
        the first of which is the Yaw/Pitch/Roll representation and
        the second is the translation
    """
    rotation_matrix = er.extract_matrix_as_np_array(reg_ds)
    ypr_degrees_assume_hfs = er.decompose_matrix_order_rpy_as_ypr_degrees(rotation_matrix)

    patient_position = plan_ds.PatientSetupSequence[0].PatientPosition

    ypr_degrees = convert_dicom_patient_ypr_to_iec_ypr(ypr_degrees_assume_hfs, patient_position)
    # ypr_dict = {"Yaw": ypr_degrees[0], "Pitch": ypr_degrees[1], "Roll": ypr_degrees[2]}

    # print(f"IEC: Yaw : Z-Rot, Pitch : X-Rot, Roll : Y-Rot")

    setup_iso_dicom_patient = np.array(ertss.extract_rtss_setup_isocenter(rtss_ds))

    print(f"Setup Isocenter (In Room): {setup_iso_dicom_patient}")
    plan_iso_dicom_patient = np.array(ep.extract_plan_setupbeam_isocenter(plan_ds))

    setup_couch_angle = plan_ds.IonBeamSequence[0].IonControlPointSequence[0].PatientSupportAngle
    print(f"Plan Isocenter (Reference): {plan_iso_dicom_patient}")
    print(f"Patient Position: {patient_position}")
    print(f"Patient Support Angle: {setup_couch_angle}")

    four_by_four_matrix = er.extract_4x4_matrix_as_np_array(reg_ds)

    rotation_inverse = rotation_matrix.transpose()  # nice feature of rotation matrices

    reg_translation = four_by_four_matrix[0:3, 3]
    print(f"Registration Translation: {reg_translation}")
    print("4x4 matrix:")
    print(f"{four_by_four_matrix}")
    delta_plan = np.array([0.0, 0.0, 0.0])
    # delta_plan = plan_iso_dicom_patient - reg_translation
    delta_plan[0] = plan_iso_dicom_patient[0] - reg_translation[0]
    delta_plan[1] = plan_iso_dicom_patient[1] - reg_translation[1]
    delta_plan[2] = plan_iso_dicom_patient[2] - reg_translation[2]

    # delta_setup = np.array([0.0, 0.0, 0.0])
    # delta_setup[0] = setup_iso_dicom_patient[0] - reg_translation[0]
    # delta_setup[1] = setup_iso_dicom_patient[1] - reg_translation[1]
    # delta_setup[2] = setup_iso_dicom_patient[2] - reg_translation[2]

    print(f"Plan - Registration Translation Vector: {delta_plan}")
    rotated_delta_plan = rotation_inverse.dot(delta_plan)
    print(f"Plan - Rotated (In Room Patient FoR) Registration Translation Vector: {rotated_delta_plan}")

    translate_dicom_patient = np.array([0.0, 0.0, 0.0])
    # translate_dicom_patient = setup_iso_dicom_patient - rotated_delta_plan
    translate_dicom_patient[0] = setup_iso_dicom_patient[0] - rotated_delta_plan[0]
    translate_dicom_patient[1] = setup_iso_dicom_patient[1] - rotated_delta_plan[1]
    translate_dicom_patient[2] = setup_iso_dicom_patient[2] - rotated_delta_plan[2]
    # test code... not sure why the table top vertical is different when Prone
    if patient_position in ["HFP", "FFP"]:
        print(f"Testing AP sign change when patient is in position: {patient_position}")
        translate_dicom_patient[1] = setup_iso_dicom_patient[1] + rotated_delta_plan[1]
        print(f"Testing Lateral sign change when patient is in position: {patient_position}")
        translate_dicom_patient[0] = setup_iso_dicom_patient[0] + rotated_delta_plan[0]

    # if (patient_position in [ "FFP", "FFS"]):
    #     print(f"Testing SupInf sign change when patient is in position: {patient_position}")
    #     translate_dicom_patient[2] = setup_iso_dicom_patient[2] + rotated_delta_plan[2]

    translate_dicom_patient_plan_frame = rotation_matrix.dot(translate_dicom_patient)
    print(f"Translation in Plan FoR: {translate_dicom_patient_plan_frame}")
    # print(f"Translation twice rotated: {rotation_inverse.dot(translate_dicom_patient)}")
    translate_iec = convert_dicom_patient_to_iec(translate_dicom_patient, patient_position)

    # xfm_setup_iso = four_by_four_matrix.dot(extend3d_to_4d(setup_iso_in_tait_bryan))
    # xfm_plan_iso = four_by_four_matrix.dot(extend3d_to_4d(plan_iso_in_tait_bryan))
    # #print(xfm_setup_iso)
    # translation_in_tait_bryan=xfm_setup_iso[0:3]-plan_iso_in_tait_bryan
    # alt_translation_in_tait_bryan= setup_iso_in_tait_bryan - xfm_plan_iso[0:3]
    # alt_translation = convert_tait_bryan_to_iec(alt_translation_in_tait_bryan)[0:3]
    # translation= convert_tait_bryan_to_iec(translation_in_tait_bryan)[0:3]
    # print(f"Alt IEC Translation: {alt_translation}")

    return ypr_degrees, translate_iec


def convert_dicom_patient_ypr_to_iec_ypr(ypr_in_dcm: np.ndarray, patient_position: str) -> np.ndarray:
    """_summary_

    Args:
        ypr_in_dcm (np.ndarray): the yaw, pitch, and roll decomposed from the SRO 4x4 in RPY order,
        but without addressing whether the patient was in some position other than HFS

        patient_position (str): The string from DICOM Patient Position (0018,5100) element, e.g. HFS HFP FFP FFS

    Returns:
        np.ndarray: the yaw, pitch, and roll that is directly applicable to an IEC 61217 Table Top
    """
    ypr_in_iec = np.array([0.0, 0.0, 0.0])
    if patient_position == "HFS":
        ypr_in_iec[0] = ypr_in_dcm[0]  #  Yaw
        ypr_in_iec[1] = ypr_in_dcm[1]  #  Pitch
        ypr_in_iec[2] = ypr_in_dcm[2]  #  Roll
    elif patient_position == "HFP":
        ypr_in_iec[0] = -ypr_in_dcm[0]  #  Z axis
        ypr_in_iec[1] = -ypr_in_dcm[1]  #  X axis
        ypr_in_iec[2] = ypr_in_dcm[2]  #  Y axis
    elif patient_position == "FFP":
        ypr_in_iec[0] = -ypr_in_dcm[0]
        ypr_in_iec[1] = ypr_in_dcm[1]
        ypr_in_iec[2] = -ypr_in_dcm[2]
    elif patient_position == "FFS":
        ypr_in_iec[0] = ypr_in_dcm[0]
        ypr_in_iec[1] = -ypr_in_dcm[1]
        ypr_in_iec[2] = -ypr_in_dcm[2]
    else:
        raise ValueError(f"patient position {patient_position} not supported yet")

    return ypr_in_iec


def convert_dicom_patient_to_tait_bryan(iso_in_dcm: np.ndarray) -> np.ndarray:
    """Convert from DICOM Patient coordinate frame to Tait-Bryan

    Args:
        iso_in_dcm (np.ndarray): the isocenter in DICOM Patient coordinates

    Returns:
        np.ndarray: the isocenter in Tait-Bryan coordinates
    """
    iso_in_tait_bryan = np.array([0.0, 0.0, 0.0])
    iso_in_tait_bryan[0] = -iso_in_dcm[2]
    iso_in_tait_bryan[1] = iso_in_dcm[0]
    iso_in_tait_bryan[2] = iso_in_dcm[1]
    return iso_in_tait_bryan


def convert_tait_bryan_to_iec(iso_in_tb: np.ndarray) -> np.ndarray:
    """Convert from Tait-Bryan coordinate frame to IEC 61217 Table Top

    Args:
        iso_in_tb (np.ndarray): isocenter in Tait-Bryan

    Returns:
        np.ndarray: isocenter in IEC 61217 Table Top
    """
    iso_in_iec = np.array([0.0, 0.0, 0.0])
    iso_in_iec[0] = iso_in_tb[1]
    iso_in_iec[1] = iso_in_tb[0]
    iso_in_iec[2] = -iso_in_tb[2]
    return iso_in_iec


def convert_dicom_patient_to_iec(iso_in_dcm: np.ndarray, patient_position: str) -> np.ndarray:
    """Convert from DICOM Patient coordinates to IEC 61217 Table Top,
    which has to take in to account the Patient Position (e.g. HFS, FFP)

    Args:
        iso_in_dcm (np.ndarray): The translation in DICOM Patient coordinates
        patient_position (str): The DICOM Patient Position coded string

    Raises:
        ValueError: When the DICOM Patient Position is not yet supported in this function

    Returns:
        np.ndarray: The translation in IEC 61217 Table Top coordinates
    """
    iso_in_iec = np.array([0.0, 0.0, 0.0])
    if patient_position == "HFS":
        iso_in_iec[0] = iso_in_dcm[0]
        iso_in_iec[1] = iso_in_dcm[2]
        iso_in_iec[2] = -iso_in_dcm[1]
    elif patient_position == "HFP":
        iso_in_iec[0] = -iso_in_dcm[0]
        iso_in_iec[1] = iso_in_dcm[2]
        iso_in_iec[2] = iso_in_dcm[1]
    elif patient_position == "FFP":
        iso_in_iec[0] = -iso_in_dcm[0]
        iso_in_iec[1] = -iso_in_dcm[2]
        iso_in_iec[2] = iso_in_dcm[1]
    elif patient_position == "FFS":
        iso_in_iec[0] = -iso_in_dcm[0]
        iso_in_iec[1] = -iso_in_dcm[2]
        iso_in_iec[2] = -iso_in_dcm[1]
    else:
        raise ValueError(f"patient position {patient_position} not supported yet")
    return iso_in_iec


def extend3d_to_4d(vec3: np.ndarray) -> np.ndarray:
    """Utility to add a fourth element for 4x4 calculations

    Args:
        vec3 (np.ndarray): a 3d vector

    Returns:
        np.ndarray: a 4d vector with 1 as the value of the fourth element
    """
    vec4 = np.array([0, 0, 0, 1], dtype=vec3.dtype)
    vec4[0] = vec3[0]
    vec4[1] = vec3[1]
    vec4[2] = vec3[2]
    return vec4


def do_calculate(sro_path:str, rtss_path:str, ionPlan_path:str):
    """Do the calculation based on the input DICOM files

    Args:
        SRO file path
        in-room RTSS file path
        RT Ion Plan file path
    """
    sro_ds = pydicom.dcmread(sro_path, force=True)
    inroom_rtss_ds = pydicom.dcmread(rtss_path, force=True)
    rtionplan_ds = pydicom.dcmread(ionPlan_path, force=True)
    ypr, translation = compute_6dof_from_reg_rtss_plan(sro_ds, inroom_rtss_ds, rtionplan_ds)
    print(f"IEC Translation in mm[Lateral, Longitudinal, Vertical]: {translation}")
    print(f"IEC Rotation [Yaw, Pitch, Roll]: {ypr}")

    print("MOSAIQ Display:")
    print(
        f"IEC Translation in cm [Lateral, Longitudinal, Vertical]:\
              [{round(translation[0]/10.0,1)}, {round(translation[1]/10.0,1)}, {round(translation[2]/10.0,1)}]"
    )
    print(f"IEC Rotation [X axis, Y axis, Z axis]: [{round(ypr[1],1)}, {round(ypr[2],1)}, {round(ypr[0],1)}]")


if __name__ == "__main__":
    do_calculate(sys.argv[1], sys.argv[2], sys.argv[3])
