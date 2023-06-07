# rtregistrationcalc
Calculates the 6DOF in IEC 61217 based on SRO, RTSS from in room imaging, and Plan

The algorithm for Table Top Corrections calculation (for MOSAIQ appears to be):

Apply the inverse rotation of the registration matrix to the difference of 
the planned isocenter (in reference CT coordinates) and the translation specified in the 4x4 registration matrix.
Subtract that from the SetupIsocenter
Convert from DICOM Patient coordinates to IEC 61217 Table Top Coordinates:

Code from compute_6dof_from_reg_rtss_plan.py:

    rotation_inverse = rotation_matrix.transpose()  # nice feature of rotation matrices
    reg_translation = four_by_four_matrix[0:3, 3]
    delta_plan = plan_iso_dicom_patient - reg_translation
    translate_dicom_patient = setup_iso_dicom_patient - rotation_inverse.dot(delta_plan)
    translate_iec = convert_dicom_patient_to_iec(translate_dicom_patient, patient_position)

The order of decomposition is driven by the rotation_matrix_to_euler_angles() function containg the following code:

        _x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        _y = math.atan2(-rotation_matrix[2, 0], _sy)
        _z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

To specify a different order of decomposition, one must generate the analytic representation (sin() and cos() for each rotation angle) and multiply in the desired order, then extract the original angles by determining which elements in combination represent the tangent of said Euler Angles.
