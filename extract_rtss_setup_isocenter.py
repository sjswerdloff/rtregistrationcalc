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

"""Extracts the SetupIsocenter from the RTSS of the in-room volumetric image

Returns:
    list[str]: coordinate as list of decimal strings
"""

import sys

import pydicom
from pydicom import Dataset


def extract_rtss_setup_isocenter(_ds: Dataset) -> list[str]:
    """Extract the isocenter value for the isocenter named "SetupIsocenter"

    Args:
        ds (Dataset): dataset representing the RT SS for the in room CT/CBCT

    Returns:
        list[str]: The isocenter value for "SetupIsocenter"
    """
    _rt_ss_iso = []
    for ss_roi_seq_item in _ds.StructureSetROISequence:
        if ss_roi_seq_item.ROIName == "SetupIsocenter":
            roi_number = ss_roi_seq_item.ROINumber
            break

    for roi_contour_seq_item in _ds.ROIContourSequence:
        if roi_contour_seq_item.ReferencedROINumber == roi_number:
            _rt_ss_iso = roi_contour_seq_item.ContourSequence[0].ContourData

    if len(_rt_ss_iso) == 0:
        raise ValueError("No SetupIsocenter")

    return _rt_ss_iso


if __name__ == "__main__":
    RTSS_PATH = sys.argv[1]
    # print(path)
    rtss_ds = pydicom.dcmread(RTSS_PATH, force=True)
    rt_ss_iso = extract_rtss_setup_isocenter(rtss_ds)
    print(rt_ss_iso)
