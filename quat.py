import numpy as np
import math
import sys
import time
import gzip
import matrix

sys.path.append('./fkf')
import fkf
import quaternion as quat

params = {
    "u": [
        0.12201555,
        0.05916921,
        -0.1526812,
        0,
        0.115313,
        -0.12255795,
        0,
        0,
        0.129171
    ],
    "offset_g": [-1.537884508, 1.615178566, 1.652199877],
    "offset_m": [-17.758405, 15.83825, -16.53175],
    "offset_a": [-0.011641857,0.011546755,-0.0218972975],
    "sigma": [0.000030461741978670858, 0.003, 0.5]
}

fast_kf = fkf.Filter(300, params['sigma'][0], params['sigma'][1],
                          params['sigma'][2], params['u'], params['offset_m'])
q_t_1 = np.array([1, 0, 0, 0])
P_t_1 = 0.001 * np.identity(4)

def generateSensorData(items):
    l = items.split(",")
    data = {
        "acceleration": {
            "x": float(l[0]),
            "y": float(l[1]),
            "z": float(l[2])
        },
        "gyro": {
            "x": float(l[3]),
            "y": float(l[4]),
            "z": float(l[5])
        },
        "magnetism": {
            "x": float(l[6]),
            "y": float(l[7]),
            "z": float(l[8])
        },
    }
    return data


def normalize(q):
    qnorm = math.sqrt((q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2))

    q[0] /= qnorm
    q[1] /= qnorm
    q[2] /= qnorm
    q[3] /= qnorm

    return q


def conjugate(q):
    return [q[0], -q[1], -q[2], -q[3]]


def inverse(q):
    qnorm = math.sqrt((q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2))
    uq = conjugate(q)

    uq[0] /= qnorm
    uq[1] /= qnorm
    uq[2] /= qnorm
    uq[3] /= qnorm

    return uq

def rotate(q, vec):
    i = conjugate(q)
    p = [0, vec[0], vec[1], vec[2]]

    # [w, v] = [q] * [p] = [a, b] * [c, d]
    a = q[0]
    b = [q[1], q[2], q[3]]

    c = p[0]
    d = [p[1], p[2], p[3]]

    w = a * c - matrix.innerProduct(b, d)
    v = matrix.crossProduct(b, d)
    v = matrix.add(v, matrix.scalarProduct(a, d))
    v = matrix.add(v, matrix.scalarProduct(c, b))

    # # [s, t] = [w, v] * [i[0], [i[1], i[2], i[3]]] = [w, v] * [j, k]
    j = i[0]
    k = [i[1], i[2], i[3]]

    s = w * j - matrix.innerProduct(v, k)
    t = matrix.crossProduct(v, k)
    t = matrix.add(t, matrix.scalarProduct(w, k))
    t = matrix.add(t, matrix.scalarProduct(j, v))

    return [s, t[0], t[1], t[2]]

def gen_quat(items):
    data = generateSensorData(items)

    global q_t_1, P_t_1

    wxb = params['offset_g'][0]
    wyb = params['offset_g'][1]
    wzb = params['offset_g'][2]
    ax = float(data['acceleration']['x'] - params['offset_a'][0])
    ay = float(data['acceleration']['y'] - params['offset_a'][1])
    az = float(data['acceleration']['z'] - params['offset_a'][2])
    wx = np.radians(float(data['gyro']['x']) - wxb)
    wy = np.radians(float(data['gyro']['y']) - wyb)
    wz = np.radians(float(data['gyro']['z']) - wzb)
    mx = float(data['magnetism']['x'])
    my = float(data['magnetism']['y'])
    mz = float(data['magnetism']['z'])

    q_t, P_t, roll, pitch, yaw = fast_kf.update(
        q_t_1, P_t_1, ax, ay, az, wx, wy, wz, mx, my, mz)
    q_t = q_t / np.linalg.norm(q_t)
    q_t = np.conjugate(q_t)
    
    gravity = quat.q_rotate(q_t, [0, 0, 1])
    linear_ax = ax - gravity[0]
    linear_ay = ay + gravity[1]
    linear_az = az - gravity[2]
    # print("{}, {}, {}".format(gravity[1], gravity[2], gravity[3]))
    rotate_acc = quat.q_rotate(q_t, [ax, ay, az])

    q_t_1 = q_t
    P_t_1 = P_t

    return [rotate_acc[0], rotate_acc[1], rotate_acc[2]]