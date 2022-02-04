from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
from data_prepare import write_data
import config

# Read data
def read_data(path):
  data = []
  with open(path, "r") as f:
    lines = f.readlines()
    for line in lines:
      dic = json.loads(line)
      data.append(dic)
  print("data_length:" + str(len(data)))
  return data


def split_data(data, train_ratio, valid_ratio, random_seed):
  """Splits data into train, validation and test according to ratio."""
  train_data = []
  valid_data = []
  test_data = []
  num_dic = {}
  for label in config.labels:
    num_dic[label] = 0
  for item in data:
    for i in num_dic:
      if item[config.LABEL_NAME] == i:
        num_dic[i] += 1
  train_num_dic = {}
  valid_num_dic = {}
  for i in num_dic:
    train_num_dic[i] = int(train_ratio * num_dic[i])
    valid_num_dic[i] = int(valid_ratio * num_dic[i])
  print(num_dic)
  random.seed(random_seed)
  random.shuffle(data)
  for item in data:
    for i in num_dic:
      if item[config.LABEL_NAME] == i:
        if train_num_dic[i] > 0:
          train_data.append(item)
          train_num_dic[i] -= 1
        elif valid_num_dic[i] > 0:
          valid_data.append(item)
          valid_num_dic[i] -= 1
        else:
          test_data.append(item)
  print("train_length:" + str(len(train_data)))
  print("test_length:" + str(len(test_data)))
  return train_data, valid_data, test_data


if __name__ == "__main__":
  data = read_data("../../data/dataset")
  train_data, valid_data, test_data = \
      split_data(data, train_ratio=config.train_ratio, \
      valid_ratio=config.valid_ratio, random_seed=config.data_split_random_seed)
  write_data(train_data, "../../data/train")
  write_data(valid_data, "../../data/valid")
  write_data(test_data, "../../data/test")
