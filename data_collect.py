import serial
import sys
import time

serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)

data = []
data_new = []
while True:
  try:
    line = s.readline().decode()
    if '---start---' in line:
      print("---start---")
      data_new.clear()
    elif '---stop---' in line:
      print("---stop---")
      if len(data_new) > 0:
        print("Data saved:")
        print(data_new)
        data.append(data_new.copy())
        data_new.clear()
      print("Data Num =", len(data))
    else:
      print(line, end="")
      data_new.append(line)
  except KeyboardInterrupt:
    filename = "gesture_"+str(time.strftime("%Y%m%d%H%M%S"))+".txt"
    with open(filename, "w") as f:
      for lines in data:
        f.write("-,-,-\n")
        for line in lines:
          f.write(line)
    print("Exiting...")
    print("Save file in", filename)
    s.close()
    sys.exit()
