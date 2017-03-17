
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

# This file was modified by Gregory Luppescu and Francisco Romero for the CS 224n project.


"""Utilities for parsing Gutenberg text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
from scipy import stats
import numpy as np
import tensorflow as tf

glovePath = "glove.42B.300d.txt"
# trainFile = "ptb.train.txt"
trainFile = "guten_train.txt"
# devFile = "ptb.valid.txt"
devFile = "guten_dev.txt"
# testFile = "ptb.test.txt"
testFile = "guten_test.txt"

def _read_words(filename):
  # Words have labels attached, so we must remove the labels from them.
  allLabels = []
  with tf.gfile.GFile(filename, "r") as f:
    allWords = f.read().decode("utf-8").replace("\n", "<eos>").split()
    
    for i in range(len(allWords)):
      label = int((allWords[i])[0])
      allWords[i] = (allWords[i])[1:]
      allLabels.append(label)

    return allWords, allLabels


def _build_vocab(filename):
  data, _ = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data, labels = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id], labels


def build_embedding(word_to_id):
  dimSize = glovePath.split('.')[-2]
  dimSize = int(dimSize.strip('d'))
  embedding_matrix = np.random.uniform(size=(len(word_to_id), dimSize), \
    low=-1.0, high=1.9)

  with open(glovePath) as text:
    for line in text:
      vector_components = line.split()
      word = vector_components[0]
      word_vector = np.zeros((dimSize,))
      if word in word_to_id:
        for i in range(1,len(vector_components)):
          word_vector[i-1] = float(vector_components[i])
        embedding_matrix[word_to_id[word]] = word_vector

  return embedding_matrix


def guten_raw_data(data_path=None):
  """Load raw dataset

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to GutenIterator.
  """

  train_path = os.path.join(data_path, trainFile)
  valid_path = os.path.join(data_path, devFile)
  test_path = os.path.join(data_path, testFile)

  word_to_id = _build_vocab(train_path)
  train_data, train_labels = _file_to_word_ids(train_path, word_to_id)
  valid_data, valid_labels = _file_to_word_ids(valid_path, word_to_id)
  test_data, test_labels = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  embedding = build_embedding(word_to_id)
  return train_data, valid_data, test_data, vocabulary, embedding, \
    train_labels, valid_labels, test_labels


def guten_producer(raw_data, raw_labels, batch_size, num_steps, name=None):
  """Iterate on the raw data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from guten_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PGProducer", [raw_data, raw_labels, batch_size, \
    num_steps]):
    raw_data_size = len(raw_data)

    max_label = np.max(np.max(raw_labels))

    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    
    shrunk_labels = tf.convert_to_tensor(raw_labels, name="shrunk_labels", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])
                      
    shrunk_labels = tf.reshape(shrunk_labels[0 : batch_size * batch_len],
                              [batch_size, batch_len])
    # shrunk_labels = []
    # for bs in range(batch_size):
    #   b_len = raw_data_size // batch_size
    #   labels_mode = \
    #     stats.mode(raw_labels[bs*b_len:bs*b_len + b_len])
    #   next_labels_row = [labels_mode[0][0]]*(num_steps)
    #   print(next_labels_row)
    #   # next_labels_row = raw_labels[bs*b_len:bs*b_len + b_len]
    #   shrunk_labels.append(next_labels_row)

    # print(len(shrunk_labels), len(shrunk_labels[0]))
    # sys.exit()

    # shrunk_labels = tf.convert_to_tensor(shrunk_labels, name="shrunk_labels")

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    
    # y = shrunk_labels

    y = tf.strided_slice(shrunk_labels, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    y.set_shape([batch_size, num_steps])
    return x, y
