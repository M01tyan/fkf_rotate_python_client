import numpy as np
import math


def vec2quat(vec):
    # exp mapping
    r = vec / 2
    q = quat_exp([0, r[0], r[1], r[2]])
    return q


def quat2vec(q):
    # log mapping
    r = (2 * quat_log(q))[1:]
    return r


def quat_norm(q):
    return np.linalg.norm(q)


def quat_normalize(q):
    q_norm = np.copy(q)
    q_norm /= quat_norm(q_norm)
    return q_norm


def quat_conjugate(q):
    q_conj = np.copy(q)  # directly work on q won't change
    q_conj[1:] *= -1
    return q_conj


def quat_inverse(q):
    q_conj = quat_conjugate(q)
    q_norm = quat_norm(q)
    return q_conj / (q_norm**2)


def quat_multiply(p, q):
    p0 = p[0]
    pv = p[1:]
    q0 = q[0]
    qv = q[1:]

    r0 = p0 * q0 - np.dot(pv, qv)
    rv = p0 * qv + q0 * pv + np.cross(pv, qv)

    return np.array([r0, rv[0], rv[1], rv[2]])


def quat_exp(q):
    q0 = q[0]
    qv = q[1:]
    qvnorm = np.linalg.norm(qv)

    z0 = np.exp(q0) * np.cos(qvnorm)
    if qvnorm == 0:
        zv = np.zeros(3)
    else:
        zv = np.exp(q0) * (qv / qvnorm) * np.sin(qvnorm)
    return np.array([z0, zv[0], zv[1], zv[2]])


def quat_log(q):
    qnorm = quat_norm(q)
    q0 = q[0]
    qv = q[1:]
    qvnorm = np.linalg.norm(qv)

    z0 = np.log(qnorm)
    if qvnorm == 0:
        zv = np.zeros(3)
    else:
        zv = (qv / qvnorm) * np.arccos(q0 / qnorm)
    return np.array([z0, zv[0], zv[1], zv[2]])


def quat_avg(q_set, qt):
    n = q_set.shape[0]

    epsilon = 1E-3
    max_iter = 1000
    for t in range(max_iter):
        err_vec = np.zeros((n, 3))
        for i in range(n):
            # calc error quaternion and transform to error vector
            qi_err = quat_normalize(quat_multiply(
                q_set[i, :], quat_inverse(qt)))
            vi_err = quat2vec(qi_err)

            # restrict vector angles to (-pi, pi]
            vi_norm = np.linalg.norm(vi_err)
            if vi_norm == 0:
                err_vec[i, :] = np.zeros(3)
            else:
                err_vec[i, :] = (-np.pi + np.mod(vi_norm +
                                                 np.pi, 2 * np.pi)) / vi_norm * vi_err

        # measure derivation between estimate and real, then update estimate
        err = np.mean(err_vec, axis=0)
        qt = quat_normalize(quat_multiply(vec2quat(err), qt))

        if np.linalg.norm(err) < epsilon:
            break

    return qt, err_vec


def q_norm(q):
    value = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    q = q / math.sqrt(value)

    return q


def q_hamilton_product(q1, q2):
    # q3 = q1*q2
    q3 = np.zeros((4,), dtype=float)
    q3[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    q3[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    q3[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    q3[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

    return q3


def q_from_matrix(R):
    q0 = math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
    q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
    q3 = (R[1, 0] - R[0, 1]) / (4 * q0)

    return np.array([q0, q1, q2, q3])


def q_from_rotation_vector(scaling, rvecdeg):
    q = np.zeros((4,), dtype=float)
    fetadeg = scaling * \
        math.sqrt(rvecdeg[0]**2 + rvecdeg[1]**2 + rvecdeg[2]**2)
    fetarad = math.radians(fetadeg)

    sinhalfeta = math.sin(0.5 * fetarad)

    if fetadeg != 0:
        q[1:4] = rvecdeg * scaling * sinhalfeta / fetadeg
    fvecsq = q[1]**2 + q[2]**2 + q[3]**2
    if fvecsq < 1.0:
        q[0] = math.sqrt(1 - fvecsq)
    else:
        q[0] = 0
    return q


def q_from_euler(roll, pitch, yaw):
    R00 = math.cos(math.radians(pitch)) * math.cos(math.radians(yaw))
    R01 = -math.cos(math.radians(roll)) * math.sin(math.radians(yaw)) + math.sin(
        math.radians(roll)) * math.sin(math.radians(pitch)) * math.sin(math.radians(yaw))
    R02 = math.sin(math.radians(roll)) * math.sin(math.radians(yaw)) + math.cos(
        math.radians(roll)) * math.sin(math.radians(pitch)) * math.sin(math.radians(yaw))
    R10 = math.cos(math.radians(pitch)) * math.sin(math.radians(yaw))
    R11 = math.cos(math.radians(roll)) * math.cos(math.radians(yaw)) + math.sin(
        math.radians(roll)) * math.sin(math.radians(pitch)) * math.sin(math.radians(yaw))
    R12 = -math.sin(math.radians(roll)) * math.cos(math.radians(yaw)) + math.cos(
        math.radians(roll)) * math.sin(math.radians(pitch)) * math.sin(math.radians(yaw))
    R20 = -math.sin(math.radians(pitch))
    R21 = math.sin(math.radians(roll)) * math.cos(math.radians(pitch))
    R22 = math.cos(math.radians(roll)) * math.cos(math.radians(pitch))

    q0 = math.sqrt(1 + R00 + R11 + R22) / 2.0
    q1 = (R21 - R12) / (4 * q0)
    q2 = (R02 - R20) / (4 * q0)
    q3 = (R10 - R01) / (4 * q0)

    return np.array([q0, q1, q2, q3])


def q_to_euler(q):
    # Bank--phi,rotation about the new X-axis
    roll = math.degrees(math.atan2(
        2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1]**2 + q[2]**2)))
    # Altitude--theta,otation about the new Y-axis
    pitch = math.degrees(math.asin(2 * (q[0] * q[2] - q[3] * q[1])))
    # Heading--psi,rotation about the Z-axis
    yaw = math.degrees(math.atan2(
        2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2)))
    return roll, pitch, yaw


def q_to_matrix(q):
    R00 = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2  # homogeneous expression
    # R00 = 1 - 2*(q[2]**2 + q[3]**2) #inhomogeneous expression
    R01 = 2 * (q[1] * q[2] - q[0] * q[3])
    R02 = 2 * (q[0] * q[2] - q[1] * q[3])
    R10 = 2 * (q[1] * q[2] + q[0] * q[3])
    R11 = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2  # homogeneous expression
    # R11 = 1 - 2*(q[1]**2 + q[3]**2) #inhomogeneous expression
    R12 = 2 * (q[2] * q[3] - q[0] * q[1])
    R20 = 2 * (q[1] * q[2] - q[0] * q[2])
    R21 = 2 * (q[0] * q[1] + q[2] * q[3])
    R22 = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2  # homogeneous expression
    # R22 = 1 - 2*(q[1]**2 + q[2]**2) #inhomogeneous expression

    return np.array([[R00, R01, R02],
                     [R10, R11, R12],
                     [R20, R21, R22]])


def q_rotate(q, vec):
    R00 = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2  # homogeneous expression
    # R00 = 1 - 2*(q[2]**2 + q[3]**2) #inhomogeneous expression
    R01 = 2 * (q[1] * q[2] - q[0] * q[3])
    R02 = 2 * (q[0] * q[2] - q[1] * q[3])
    R10 = 2 * (q[1] * q[2] + q[0] * q[3])
    R11 = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2  # homogeneous expression
    # R11 = 1 - 2*(q[1]**2 + q[3]**2) #inhomogeneous expression
    R12 = 2 * (q[2] * q[3] - q[0] * q[1])
    R20 = 2 * (q[1] * q[2] - q[0] * q[2])
    R21 = 2 * (q[0] * q[1] + q[2] * q[3])
    R22 = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2  # homogeneous expression
    # R22 = 1 - 2*(q[1]**2 + q[2]**2) #inhomogeneous expression

    x = R00 * vec[0] + R01 * vec[1] + R02 * vec[2]
    y = R10 * vec[0] + R11 * vec[1] + R12 * vec[2]
    z = R20 * vec[0] + R21 * vec[1] + R22 * vec[2]

    return np.array([x, y, z])
