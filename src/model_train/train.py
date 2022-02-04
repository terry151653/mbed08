# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=redefined-outer-name
# pylint: disable=g-bad-import-order

"""Build and train neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
from data_load import DataLoader

import numpy as np
import tensorflow as tf

import config

logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

def reshape_function(data, label):
  reshaped_data = tf.reshape(data, [-1, 3, 1])
  return reshaped_data, label


def calculate_model_size(model):
  print(model.summary())
  var_sizes = [
      np.product(list(map(int, v.shape))) * v.dtype.size
      for v in model.trainable_variables
  ]
  print("Model size:", sum(var_sizes) / 1024, "KB")


def load_data(train_data_path, valid_data_path, test_data_path, seq_length):
  data_loader = DataLoader(
      train_data_path, valid_data_path, test_data_path, seq_length=seq_length)
  data_loader.format()
  return data_loader.train_len, data_loader.train_data, data_loader.valid_len, \
      data_loader.valid_data, data_loader.test_len, data_loader.test_data


def build_net(seq_length):
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          8, (4, 3),
          padding="same",
          activation="relu",
          input_shape=(seq_length, 3, 1)),
      tf.keras.layers.MaxPool2D((3, 3)),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Conv2D(16, (4, 1), padding="same",
                             activation="relu"),
      tf.keras.layers.MaxPool2D((3, 1), padding="same"),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(16, activation="relu"),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(len(config.labels), activation="softmax")
  ])
  print("Built CNN.")
  model_path = "../../model/"
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  return model, model_path


def train_net(
    model,
    model_path,
    train_len,
    train_data,
    valid_len,
    valid_data,
    test_len,
    test_data,
    kind):
  """Trains the model."""
  calculate_model_size(model)
  model.compile(
      optimizer="adam",
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"])
  if kind == "CNN":
    train_data = train_data.map(reshape_function)
    test_data = test_data.map(reshape_function)
    valid_data = valid_data.map(reshape_function)
  test_labels = np.zeros(test_len)
  idx = 0
  for _, label in test_data:
    test_labels[idx] = label.numpy()
    idx += 1
  train_data = train_data.batch(config.batch_size).repeat()
  valid_data = valid_data.batch(config.batch_size)
  test_data = test_data.batch(config.batch_size)
  model.fit(
      train_data,
      epochs=config.epochs,
      validation_data=valid_data,
      steps_per_epoch=config.steps_per_epoch,
      validation_steps=int((valid_len - 1) / config.batch_size + 1),
      callbacks=[tensorboard_callback])
  loss, acc = model.evaluate(test_data)
  pred = np.argmax(model.predict(test_data), axis=1)
  confusion = tf.math.confusion_matrix(
      labels=tf.constant(test_labels),
      predictions=tf.constant(pred),
      num_classes=len(config.labels))
  print(confusion)
  print("Loss {}, Accuracy {}".format(loss, acc))

  # Convert the model to the TensorFlow Lite format without quantization
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # Save the model to disk
  open(model_path+"model.tflite", "wb").write(tflite_model)

  # Convert the model to the TensorFlow Lite format with quantization
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
  tflite_model = converter.convert()

  # Save the model to disk
  open(model_path+"model_quantized.tflite", "wb").write(tflite_model)

  basic_model_size = os.path.getsize(model_path+"model.tflite")
  print("Basic model is %d bytes" % basic_model_size)
  quantized_model_size = os.path.getsize(model_path+"model_quantized.tflite")
  print("Quantized model is %d bytes" % quantized_model_size)
  difference = basic_model_size - quantized_model_size
  print("Difference is %d bytes" % difference)


if __name__ == "__main__":
  print("Start to load data...")
  train_len, train_data, valid_len, valid_data, test_len, test_data = \
      load_data("../../data/train", "../../data/valid", "../../data/test",\
      config.seq_length)

  print("Start to build net...")
  model, model_path = build_net(config.seq_length)

  print("Start training...")
  train_net(model, model_path, train_len, train_data, valid_len, valid_data,
            test_len, test_data, config.model)

  print("Training finished!")
