# Author: Zeyang Dai
# Date: 2018-04-07
# Email: zeyang-d@outlook.com
# Decription: Novel MARG-Sensor Orientation Estimation Algorithm Using Fast Kalman Filter, experiments show that Magnetic disturbance has no
# effect on pitch and roll, but affect yaw

import numpy as np
import math
import quaternion as quat


class Filter:
    deltaT = 0

    sigmaA2 = 0
    sigmaG2 = 0
    sigmaM2 = 0

    matrix = []
    centerOffset = []

    def __init__(self, frequency, var_acc, var_gyro, var_mag, matrix, m_offset):
        self.deltaT = 1.0 / frequency
        self.sigmaA2 = var_acc
        self.sigmaG2 = var_gyro
        self.sigmaM2 = var_mag

        self.matrix = [matrix[0:3], matrix[3:6], matrix[6:9]]
        self.centerOffset = m_offset

    # @numba.jit(nopython=True, parallel=True)
    def q_from_gyro(self, q_t_1, P_t_1, wx, wy, wz):
        omegaX = np.array([[0, -wx, -wy, -wz],
                           [wx, 0, wz, -wy],
                           [wy, -wz, 0, wx],
                           [wz, wy, -wx, 0]])
        q = np.dot((np.identity(4) + (self.deltaT / 2.0) * omegaX), q_t_1)
        Sigma_t = np.array([[q_t_1[1], q_t_1[2], q_t_1[3]],
                            [-q_t_1[0], -q_t_1[3], -q_t_1[2]],
                            [q_t_1[2], -q_t_1[0], -q_t_1[1]],
                            [-q_t_1[2], q_t_1[1], -q_t_1[0]]])
        Sigma_gyro = np.array([[self.sigmaG2, 0, 0],
                               [0, self.sigmaG2, 0],
                               [0, 0, self.sigmaG2]])
        Sigma_w_t = (self.deltaT / 2)**2 * np.dot(np.dot(Sigma_t.reshape(4,
                                                                         3), Sigma_gyro), np.transpose(Sigma_t.reshape(4, 3)))

        P_t_ = np.dot(np.dot((np.identity(4) + (self.deltaT / 2) * omegaX), P_t_1),
                      np.transpose(np.identity(4) + (self.deltaT / 2) * omegaX)) + Sigma_w_t

        return q, Sigma_w_t, P_t_

    # @numba.jit(nopython=True, parallel=True)
    def q_from_acc_mag(self, q_t_1, ax, ay, az, mx, my, mz):

        mD = ax * mx + ay * my + az * mz
        mN = math.sqrt(1 - mD**2)

        Wa = np.array([[az, ay, -ax, 0],
                       [ay, -az, 0, ax],
                       [-ax, 0, -az, ay],
                       [0, ax, ay, az]])
        Wm = np.array([[(mN * mx + mD * mz), (mD * my), (mN * mz - mD * mx), -mN * my],
                       [(mD * my), (mN * mx - mD * mz),
                        (mN * my), (mN * mz + mD * mx)],
                       [(mN * mz - mD * mx), (mN * my),
                        (-mN * mx - mD * mz), (mD * my)],
                       [-mN * my, (mN * mz + mD * mx), (mD * my), (-mN * mx + mD * mz)]])

        q = (1.0 / 4.0) * np.dot(np.dot((Wm + np.identity(4)),
                                        (Wa + np.identity(4))), q_t_1)

        q = q / np.linalg.norm(q)

        J11 = (-q_t_1[2] - mN * (mz * q_t_1[0] + my * q_t_1[1] - mx * q_t_1[2]) +
               mD * (mx * q_t_1[0] + mz * q_t_1[2] - my * q_t_1[3])) / 4
        J12 = (q_t_1[1] + mN * mx * q_t_1[1] + mN * my * q_t_1[2] + mN * mz *
               q_t_1[3] + mD * (my * q_t_1[0] - mz * q_t_1[1] + mx * q_t_1[3])) / 4
        J13 = (q_t_1[0] + mN * mx * q_t_1[0] + mD * mz * q_t_1[0] + mD * my *
               q_t_1[1] - mD * mx * q_t_1[2] + mN * mz * q_t_1[2] - mN * my * q_t_1[3]) / 4
        J14 = ((ax * mD + mN + az * mN) * q_t_1[0] + ay * mN *
               q_t_1[1] + (-((1 + az) * mD) + ax * mN) * q_t_1[2]) / 4
        J15 = (ay * mD * q_t_1[0] + (mD + az * mD - ax * mN) * q_t_1[1] +
               ay * mN * q_t_1[2] - (ax * mD + mN + az * mN) * q_t_1[3]) / 4
        J16 = (mD * (q_t_1[0] + az * q_t_1[0] - ay * q_t_1[1] + ax * q_t_1[2]) +
               mN * (-(ax * q_t_1[0]) + q_t_1[2] + az * q_t_1[2] + ay * q_t_1[3])) / 4
        J21 = (q_t_1[3] - mN * (my * q_t_1[0] - mz * q_t_1[1] + mx * q_t_1[3]) +
               mD * (mx * q_t_1[1] + my * q_t_1[2] + mz * q_t_1[3])) / 4
        J22 = (q_t_1[0] + mN * mx * q_t_1[0] + mD * mz * q_t_1[0] + mD * my *
               q_t_1[1] - mD * mx * q_t_1[2] + mN * mz * q_t_1[2] - mN * my * q_t_1[3]) / 4
        J23 = (-((1 + mN * mx) * q_t_1[1]) - mD * (my * q_t_1[0] - mz *
                                                   q_t_1[1] + mx * q_t_1[3]) - mN * (my * q_t_1[2] + mz * q_t_1[3])) / 4
        J24 = (ay * (mN * q_t_1[0] - mD * q_t_1[2]) - (-1 + az) * (mN *
                                                                   q_t_1[1] + mD * q_t_1[3]) + ax * (mD * q_t_1[1] - mN * q_t_1[3])) / 4
        J25 = (mD * (q_t_1[0] - az * q_t_1[0] + ay * q_t_1[1] + ax * q_t_1[2]) -
               mN * (ax * q_t_1[0] + (-1 + az) * q_t_1[2] + ay * q_t_1[3])) / 4
        J26 = (ay * (mD * q_t_1[0] + mN * q_t_1[2]) + mD * ((-1 + az) * q_t_1[1] +
                                                            ax * q_t_1[3]) + mN * (ax * q_t_1[1] + q_t_1[3] - az * q_t_1[3])) / 4
        J31 = (-((1 + mN * mx + mD * mz) * q_t_1[0]) - mD * my * q_t_1[1] +
               mD * mx * q_t_1[2] - mN * mz * q_t_1[2] + mN * my * q_t_1[3]) / 4
        J32 = (q_t_1[3] - mN * (my * q_t_1[0] - mz * q_t_1[1] + mx * q_t_1[3]) +
               mD * (mx * q_t_1[1] + my * q_t_1[2] + mz * q_t_1[3])) / 4
        J33 = (-q_t_1[2] - mN * (mz * q_t_1[0] + my * q_t_1[1] - mx * q_t_1[2]) +
               mD * (mx * q_t_1[0] + mz * q_t_1[2] - my * q_t_1[3])) / 4
        J34 = (mD * ((-1 + az) * q_t_1[0] + ay * q_t_1[1] + ax * q_t_1[2]) -
               mN * (ax * q_t_1[0] + q_t_1[2] - az * q_t_1[2] + ay * q_t_1[3])) / 4
        J35 = (ay * (-(mN * q_t_1[0]) + mD * q_t_1[2]) - (-1 + az) * (mN *
                                                                      q_t_1[1] + mD * q_t_1[3]) + ax * (-(mD * q_t_1[1]) + mN * q_t_1[3])) / 4
        J36 = (mN * (q_t_1[0] - az * q_t_1[0] + ay * q_t_1[1]) - ax * (mD *
                                                                       q_t_1[0] + mN * q_t_1[2]) + mD * ((-1 + az) * q_t_1[2] + ay * q_t_1[3])) / 4
        J41 = (q_t_1[1] + mN * mx * q_t_1[1] + mN * my * q_t_1[2] + mN * mz *
               q_t_1[3] + mD * (my * q_t_1[0] - mz * q_t_1[1] + mx * q_t_1[3])) / 4
        J42 = (q_t_1[2] + mN * (mz * q_t_1[0] + my * q_t_1[1] - mx * q_t_1[2]) -
               mD * (mx * q_t_1[0] + mz * q_t_1[2] - my * q_t_1[3])) / 4
        J43 = (q_t_1[3] - mN * (my * q_t_1[0] - mz * q_t_1[1] + mx * q_t_1[3]) +
               mD * (mx * q_t_1[1] + my * q_t_1[2] + mz * q_t_1[3])) / 4
        J44 = (-(ay * (mD * q_t_1[0] + mN * q_t_1[2])) + ax * (mN * q_t_1[1] +
                                                               mD * q_t_1[3]) + (1 + az) * (mD * q_t_1[1] - mN * q_t_1[3])) / 4
        J45 = ((1 + az) * (-(mN * q_t_1[0]) + mD * q_t_1[2]) + ax * (mD *
                                                                     q_t_1[0] + mN * q_t_1[2]) + ay * (mN * q_t_1[1] + mD * q_t_1[3])) / 4
        J46 = (ay * (mN * q_t_1[0] - mD * q_t_1[2]) + (1 + az) * (mN * q_t_1[1] +
                                                                  mD * q_t_1[3]) + ax * (-(mD * q_t_1[1] + mN * q_t_1[3]))) / 4

        J = np.array([[J11, J12, J13, J14, J15, J16],
                      [J21, J22, J23, J24, J25, J26],
                      [J31, J32, J33, J34, J35, J36],
                      [J41, J42, J43, J44, J45, J46]])
        Epsilon_acc_mag = np.array([[self.sigmaA2, 0, 0, 0, 0, 0],
                                    [0, self.sigmaA2, 0, 0, 0, 0],
                                    [0, 0, self.sigmaA2, 0, 0, 0],
                                    [0, 0, 0, self.sigmaM2, 0, 0],
                                    [0, 0, 0, 0, self.sigmaM2, 0],
                                    [0, 0, 0, 0, 0, self.sigmaM2]])

        Epsilon_v = np.dot(
            np.dot(J.reshape(4, 6), Epsilon_acc_mag), np.transpose(J.reshape(4, 6)))

        return q, Epsilon_v

    def update(self, q_t_1, P_t_1, ax, ay, az, wx, wy, wz, mx, my, mz):
        u = np.array(self.matrix)
        c = np.array(self.centerOffset)

        # normalization
        acc_norm = np.linalg.norm([ax, ay, az])
        ax = ax / acc_norm
        ay = ay / acc_norm
        az = az / acc_norm
        wx = float(wx)
        wy = float(wy)
        wz = float(wz)
        [mx, my, mz] = u.dot([mx, my, mz] - c)  # correct method
        mag_norm = np.linalg.norm([mx, my, mz])
        mx = float(mx / mag_norm)
        my = float(my / mag_norm)
        mz = float(mz / mag_norm)

        # prediction
        q_t_, Sigma_w_t, P_t_ = self.q_from_gyro(q_t_1, P_t_1, wx, wy, wz)

        # measurement
        z_t, Sigma_v_t = self.q_from_acc_mag(q_t_1, ax, ay, az, mx, my, mz)

        # kalman gain calculation
        K_t = np.dot(P_t_, np.linalg.inv(P_t_ + Sigma_v_t))

        # update
        q_t = q_t_ + np.dot(K_t, (z_t - q_t_))
        P_t = np.dot((np.identity(4) - K_t), P_t_)

        q_t_norm = np.linalg.norm(q_t)
        q_t[0] = float(q_t[0] / q_t_norm)
        q_t[1] = float(q_t[1] / q_t_norm)
        q_t[2] = float(q_t[2] / q_t_norm)
        q_t[3] = float(q_t[3] / q_t_norm)

        roll, pitch, yaw = quat.q_to_euler(q_t)

        return q_t, P_t, roll, pitch, yaw
