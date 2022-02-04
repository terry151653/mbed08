from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import os
import random
import config

def prepare_data(label, data, file_to_read):
  """Read collected data from files."""
  with open(file_to_read, "r") as f:
    lines = csv.reader(f)
    data_new = {}
    data_new[config.LABEL_NAME] = label
    data_new[config.DATA_NAME] = []
    for line in lines:
      if len(line) == 3:
        if line[2] == "-" and data_new[config.DATA_NAME]:
          data.append(data_new)
          data_new = {}
          data_new[config.LABEL_NAME] = label
          data_new[config.DATA_NAME] = []
        elif line[2] != "-":
          data_new[config.DATA_NAME].append([float(i) for i in line[0:3]])
    data.append(data_new)


def generate_negative_data(data):
  """Generate negative data."""

  # Big movement -> around straight line
  for i in range(100):
    dic = {config.DATA_NAME: [], config.LABEL_NAME: "negative"}
    start_x = (random.random() - 0.5) * 2000
    start_y = (random.random() - 0.5) * 2000
    start_z = (random.random() - 0.5) * 2000
    x_increase = (random.random() - 0.5) * 10
    y_increase = (random.random() - 0.5) * 10
    z_increase = (random.random() - 0.5) * 10
    for j in range(config.seq_length):
      dic[config.DATA_NAME].append([
          start_x + j * x_increase + (random.random() - 0.5) * 6,
          start_y + j * y_increase + (random.random() - 0.5) * 6,
          start_z + j * z_increase + (random.random() - 0.5) * 6
      ])
    data.append(dic)

  # Random
  for i in range(100):
    dic = {config.DATA_NAME: [], config.LABEL_NAME: "negative"}
    for _ in range(config.seq_length):
      dic[config.DATA_NAME].append([(random.random() - 0.5) * 1000,
                             (random.random() - 0.5) * 1000,
                             (random.random() - 0.5) * 1000])
    data.append(dic)

  # Stay still
  for i in range(100):
    dic = {config.DATA_NAME: [], config.LABEL_NAME: "negative"}
    start_x = (random.random() - 0.5) * 2000
    start_y = (random.random() - 0.5) * 2000
    start_z = (random.random() - 0.5) * 2000
    for _ in range(config.seq_length):
      dic[config.DATA_NAME].append([
          start_x + (random.random() - 0.5) * 40,
          start_y + (random.random() - 0.5) * 40,
          start_z + (random.random() - 0.5) * 40
      ])
    data.append(dic)


# Write data to file
def write_data(data_to_write, path):
  with open(path, "w") as f:
    for item in data_to_write:
      dic = json.dumps(item, ensure_ascii=False)
      f.write(dic)
      f.write("\n")

if __name__ == "__main__":
  data = []
  for label in config.labels:
    prepare_data(label, data, "../../data/raw/%s_%s.txt" % (config.LABEL_NAME, label))
  generate_negative_data(data)
  print("data_length: " + str(len(data)))

  write_data(data, "../../data/dataset")
