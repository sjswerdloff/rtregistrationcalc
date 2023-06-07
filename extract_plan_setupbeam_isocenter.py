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

"""Extract the isocenter from the first beam in the plan, presumed to be a SETUP beam

Returns:
    list[str]: the isocenter as a list of decimal string
"""

import sys

import pydicom


def extract_plan_setupbeam_isocenter(_ds: pydicom.Dataset) -> list[str]:
    """Extracts the isocenter from the first beam in the plan

    Args:
        ds (pydicom.Dataset): dataset representing the plan

    Returns:
        list[str]: the isocenter in the first beam
    """
    plan_setup_iso = []
    plan_setup_iso = _ds.IonBeamSequence[0].IonControlPointSequence[0].IsocenterPosition
    if len(plan_setup_iso) == 0:
        raise ValueError("No isocenter in first beam of plan")

    return plan_setup_iso


if __name__ == "__main__":
    PLAN_PATH = sys.argv[1]
    # print(path)
    plan_ds = pydicom.dcmread(PLAN_PATH, force=True)
    plan_iso = extract_plan_setupbeam_isocenter(plan_ds)
    print(plan_iso)
