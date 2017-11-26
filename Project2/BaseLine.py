#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import csv
from localPath import *
from loadData2 import trainingData, trainingLabel, valData, valLabel, picSize, testData

tf.logging.set_verbosity(tf.logging.INFO)



def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [ , width, height, channels]
  input_layer = tf.reshape(features["x"], [-1, 1, picSize[0], picSize[1], 3])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, picSize[0], picSize[1], 1]
  # Output Tensor Shape: [batch_size, picSize[0], picSize[1], 64]
  conv1FilterNumbers = 32
  conv1 = tf.layers.conv3d(
      inputs=input_layer,
      filters=conv1FilterNumbers,
      kernel_size=[3, 5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, picSize[0], picSize[1], 32]
  # Output Tensor Shape: [batch_size, 50, 50, 32]
  # (N - F) / stride + 1
  # (50 - 2) / 2 + 1 = 25
  conv1_1 = tf.reshape(conv1, [-1, picSize[0], picSize[1], conv1FilterNumbers])
  pool1 = tf.layers.max_pooling2d(inputs=conv1_1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 25, 25, 32]
  # Output Tensor Shape: [batch_size, 25, 25, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 25, 25, 64]
  # Output Tensor Shape: [batch_size, 25, 25, 64]
  # (N - F) / stride + 1
  # (25 - 3) / 2 + 1 = 12
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 12, 12, 64]
  # Output Tensor Shape: [batch_size, 5 * 5 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 12 * 12 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 25 * 25 * 128]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.6, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 5]
  logits = tf.layers.dense(inputs=dropout, units=5)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=5)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  train_data = trainingData
  train_labels = trainingLabel
  eval_data = valData
  eval_labels = valLabel

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=logPath2)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=1000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

  # prediction
  pre_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": testData},
      num_epochs=1,
      shuffle=False)
  pre_results = mnist_classifier.predict(input_fn=pre_input_fn)
  resultlist = []
  for i, p in enumerate(pre_results):
      resultlist.append(p['classes'])

  csvFile = open(dataPath + 'testlabel.csv', "w")
  writer = csv.writer(csvFile)
  for e in resultlist:
      writer.writerow([int(e)])
  csvFile.close()


if __name__ == "__main__":
  tf.app.run()