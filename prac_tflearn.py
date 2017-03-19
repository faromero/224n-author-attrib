#! /usr/bin/python

# -*- coding: utf-8 -*-
"""
Using LSTM recurrent neural network to classify authors
"""
from __future__ import division, print_function, absolute_import

import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.metrics import confusion_matrix
from tflearn.datasets import imdb
from basic_classif import makeData

###################################################################
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    np.set_printoptions(precision=2)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      if len(str(cm[i,j])) > 4:
        cm[i,j] = float(str(cm[i,j])[0:4])

      plt.text(j, i, cm[i, j],
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
###################################################################

trainX, trainY, validX, validY, testX, testY = makeData(window_size=200)

trainX = np.array(trainX)
trainY = np.array(trainY)
validX = np.array(validX)
validY = np.array(validY)
testX = np.array(testX)
testY = np.array(testY)

print("Partitioned size of data for training")
print(trainX.shape)
print(trainY.shape)

print("Partitioned size of data for valid")
print(validX.shape)
print(validY.shape)

print("Partitioned size of data for testing")
print(testX.shape)
print(testY.shape)

maxInd = np.max(np.max(trainX))

embSize = 200

# Network building
net = tflearn.input_data([None, (trainX.shape)[1]])
net = tflearn.embedding(net, input_dim=maxInd+1, trainable=False, output_dim=embSize, name="EmbeddingLayer")
net = tflearn.lstm(net, embSize, dropout=0.8)
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(validX, validY), show_metric=True,
          batch_size=100,  n_epoch=1)

predictions = model.predict(testX)

accuracy = np.mean(np.argmax(testY, axis=1) == np.argmax(predictions, axis=1))
print('Test Accuracy:', accuracy)

# Compute confusion matrix
cnf_matrix = confusion_matrix(np.argmax(testY, axis=1), \
  np.argmax(predictions, axis=1))

print(cnf_matrix)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
# class_names = ['Charles Darwin', 'Edgar Allan Poe', 'Edward Stratemeyer',\
#     'Jacob Abbott', 'Lewis Carroll','Mark Twain',\
#      'Michael Faraday', 'Ralph Waldo Emerson', \
#      'Rudyard Kipling', 'Winston Churchill']
class_names = ['CD', 'EAP', 'ES', 'JA', 'LC','MT', \
               'MF', 'RWE', 'RK', 'WC']
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                title='LSTM Classifier Confusion Matrix')

plt.show()

