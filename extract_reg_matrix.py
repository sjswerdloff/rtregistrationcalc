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

"""Extracts the transformation matrix from an SRO

Returns:
    np.ndarray: various size matrices
"""

import math
import sys

import numpy as np
import pydicom

import convert_matrix_to_euler as cnv


def extract_matrix_as_np_array(sro_ds: pydicom.Dataset) -> np.ndarray:
    """Extract the 3x3 rotation matrix from the first registration matrix in the provides Spatial Registration Object

    Args:
        ds (pydicom.Dataset): Dataset representing the Spatial Registration Object

    Returns:
        np.ndarray: The 3x3 rotation matrix as a numpy array
    """
    matrix = (
        sro_ds.RegistrationSequence[0].MatrixRegistrationSequence[0].MatrixSequence[0].FrameOfReferenceTransformationMatrix
    )
    row0 = matrix[0:3]
    # print(row0)
    row1 = matrix[4:7]
    # print(row1)
    row2 = matrix[8:11]
    # print(row2)
    # print(matrix)
    rotation_mtx = np.array([row0, row1, row2])
    return rotation_mtx


def extract_4x4_matrix_as_np_array(sro_ds: pydicom.Dataset) -> np.ndarray:
    """Extract the 4x4 transformation matrix from the first registration matrix in the provides Spatial Registration Object
        One can extract subsets, such as the 3x3 rotation matrix and 1x3 translation vector by slicing the value returned
    Args:
        ds (pydicom.Dataset): Dataset representing the Spatial Registration Object

    Returns:
        np.ndarray: The 4x4 transformation matrix as a numpy array
    """
    matrix = (
        sro_ds.RegistrationSequence[0].MatrixRegistrationSequence[0].MatrixSequence[0].FrameOfReferenceTransformationMatrix
    )
    row0 = matrix[0:4]
    # print(row0)
    row1 = matrix[4:8]
    # print(row1)
    row2 = matrix[8:12]
    # print(row2)
    row3 = matrix[12:16]
    # print(matrix)
    transform_mtx = np.array([row0, row1, row2, row3])
    return transform_mtx


def decompose_matrix_order_rpy_as_ypr_degrees(rotation_mtx: np.ndarray, tolerance_ortho_normality: float | None = None) -> np.ndarray:
    """Decomposes the provided 3x3 matrix into Yaw, Pitch, and Roll
    The decomposition order is Roll, Pitch, Yaw (because IEC 61217 and DICOM state that the application of the values
    is to be performed translation first, then yaw, then pitch, then roll, so the decomposition reverses that

    Args:
        R (np.ndarray): the 3x3 rotation matrix

    Returns:
        np.ndarray: the IEC 61217 Table Top rotation angles (with Patient Support Angle being Yaw)
    """
    euler_angles = cnv.rotation_matrix_to_euler_angles(rotation_mtx, tolerance_ortho_normality=tolerance_ortho_normality)
    # Rprime = cnv.eulerAnglesToRotationMatrix(euler_angles)
    in_degrees = euler_angles * 180.0 / math.pi
    _iec_pitch = in_degrees[0]
    _iec_roll = in_degrees[2]
    _iec_yaw = -in_degrees[1]
    return np.array([_iec_yaw, _iec_pitch, _iec_roll])


if __name__ == "__main__":
    SRO_PATH = sys.argv[1]
    # print(path)
    reg_ds = pydicom.dcmread(SRO_PATH, force=True)
    # matrix = ds.RegistrationSequence[0].MatrixRegistrationSequence[0].MatrixSequence[0].FrameOfReferenceTransformationMatrix
    rotation_matrix = extract_matrix_as_np_array(reg_ds)
    iec_angles = decompose_matrix_order_rpy_as_ypr_degrees(rotation_matrix)
    iec_yaw = iec_angles[0]
    iec_pitch = iec_angles[1]
    iec_roll = iec_angles[2]
    print(rotation_matrix)
    # print(rotation_matrix[0])
    # print(rotation_matrix[1])
    # print(rotation_matrix[2])
    print(f"Pitch(rot-X): {iec_pitch} Roll(rot-Y): {iec_roll} Yaw(rot-z): {iec_yaw}")
    print("  Or when rounded to one digit after the decimal:")
    print(f"Pitch(rot-X): {round(iec_pitch,1)} Roll(rot-Y): {round(iec_roll,1)} Yaw(rot-z): {round(iec_yaw,1)}")
