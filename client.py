import socket
import quat
import numpy as np

cnt = 0
data = {
    "acceleration": {
        "x": [],
        "y": [],
        "z": []
    },
    "gyro": {
        "x": [],
        "y": [],
        "z": []
    },
    "magnetism": {
        "x": [],
        "y": [],
        "z": []
    },
}

def appendSensorData(items):
  l = items.split(",")
  global data
  data['acceleration']['x'].append(float(l[0]))
  data['acceleration']['y'].append(float(l[1]))
  data['acceleration']['z'].append(float(l[2]))
  data['gyro']['x'].append(float(l[3]))
  data['gyro']['y'].append(float(l[4]))
  data['gyro']['z'].append(float(l[5]))
  data['magnetism']['x'].append(float(l[6]))
  data['magnetism']['y'].append(float(l[7]))
  data['magnetism']['z'].append(float(l[8]))


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
  # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    c.connect(("192.168.11.22", 6000))
    # s.bind(("192.168.11.255", 8080))
    # s.listen(1)
    # conn, addr = s.accept()
    count = 0
    items = ""
    item = ""
    # with conn:
    while True:
      value = c.recv(1).decode('utf-8')
      if value == '\n':
        appendSensorData(items)
        if len(data['acceleration']['x']) > 30:
          data['acceleration']['x'].pop(0)
          data['acceleration']['y'].pop(0)
          data['acceleration']['z'].pop(0)
          data['gyro']['x'].pop(0)
          data['gyro']['y'].pop(0)
          data['gyro']['z'].pop(0)
          data['magnetism']['x'].pop(0)
          data['magnetism']['y'].pop(0)
          data['magnetism']['z'].pop(0)
          acc = np.array([np.average(data['acceleration']['x']), np.average(data['acceleration']['y']), np.average(data['acceleration']['z'])])
          gyro = np.array([np.average(data['gyro']['x']), np.average(data['gyro']['y']), np.average(data['gyro']['z'])])
          mag = np.array([np.average(data['magnetism']['x']), np.average(data['magnetism']['y']), np.average(data['magnetism']['z'])])
          rotate_data = quat.gen_quat("{}, {}, {}, {}, {}, {}, {}, {}, {}".format(acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2], mag[0], mag[1], mag[2]))
        # data = quat.gen_quat(items)
          if cnt > 1500:
          # send_data = "{},{},{}\n".format(data[0], data[1], data[2])
          # conn.send(send_data.encode('utf-8'))
            print("{},{},{}".format(rotate_data[0], rotate_data[1], rotate_data[2]))
        count = 0
        items = ""
        cnt += 1
      else:
        if value == ',':
          if 1 <= count and count <= 8:
            items += item + ","
          elif count == 9:
            items += item
          item = ""
          count += 1
        else:
          item += value
    server.close()
    client.close()