# -*- coding: utf-8 -*-


def innerProduct(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("v1 and v2 length is not same")

    if (len(v1) + len(v2)) != 6:
        raise ValueError("v1 or v2 length is error")

    res = 0.0
    for i in range(3):
        res += v1[i] * v2[i]

    return res


def crossProduct(v1, v2):
    if len(v1) != 3:
        raise ValueError("v1 length is error")

    if len(v2) != 3:
        raise ValueError("v2 lenght is error")

    x = v1[1] * v2[2] - v1[2] * v2[1]
    y = v1[2] * v2[0] - v1[0] * v2[2]
    z = v1[0] * v2[1] - v1[1] * v2[0]

    return [x, y, z]


def scalarProduct(scalar, v):
    if len(v) != 3:
        raise ValueError("v length is error")

    x = scalar * v[0]
    y = scalar * v[1]
    z = scalar * v[2]

    return [x, y, z]


def add(v1, v2):
    if len(v1) != 3:
        raise ValueError("v1 length is error")

    if len(v2) != 3:
        raise ValueError("v2 lenght is error")

    x = v1[0] + v2[0]
    y = v1[1] + v2[1]
    z = v1[2] + v2[2]

    return [x, y, z]


def main():
    print("- inner product test:")
    print("res:", innerProduct([1, 2, 3], [1, 2, 3]))
    print("ans:", 14)
    print()

    print("- cross product test:")
    print("res:", crossProduct([1, 2, 3], [2, 3, 4]))
    print("ans:", [-1, 2, -1])
    print()

    print("- scalar product test:")
    print("res:", scalarProduct(2, [2, 3, 4]))
    print("ans:", [4, 6, 8])
    print()

    print("- add test:")
    print("res:", add([1, 1, 1], [2, 3, 4]))
    print("ans:", [3, 4, 5])


if __name__ == '__main__':
    main()