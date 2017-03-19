#! /usr/bin/python

import time
import itertools
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
from reader_class import *

numIter = 10000

###################################################################
# def plot_confusion_matrix(cm, classes,
#                           normalize=True,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     np.set_printoptions(precision=2)
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#       if len(str(cm[i,j])) > 4:
#         cm[i,j] = float(str(cm[i,j])[0:4])
#
#       plt.text(j, i, cm[i, j],
#                horizontalalignment="center",
#                color="white" if cm[i, j] > thresh else "black")
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
###################################################################

def makeData(dataPath='clean_books_div', window_size=400):
  train_data, valid_data, test_data, _, embedding, train_labels, \
    valid_labels, test_labels = guten_raw_data(data_path=dataPath)

  print "Total size of data"
  print 'Train, Dev, Test:', len(train_data), len(valid_data), len(test_data)
  print 'Embeddings:', len(embedding), len(embedding[0])

  train_batch_len = len(train_data) // window_size
  maxLabel = np.max(np.max(train_labels))

  train_x_lu = []
  train_y_lu = []
  for x in range(train_batch_len):
    train_x_lu.append(train_data[x*window_size:x*window_size + window_size])
    labels_mode = \
      stats.mode(train_labels[x*window_size:x*window_size + window_size])
    next_labels_row = [0]*(maxLabel+1)
    next_labels_row[labels_mode[0][0]] = 1
    train_y_lu.append(next_labels_row)
    

  train_x = np.zeros(shape=(len(train_x_lu), len(embedding[0])))
  train_y = np.array(train_y_lu)

  for i in range(len(train_x_lu)):
    currRow = np.zeros(shape=(len(embedding[0]), ))
    for j in range(len(train_x_lu[0])):
      nextEmb = np.array(embedding[train_x_lu[i][j]])
      # print nextEmb.shape, currRow.shape
      currRow += nextEmb

    train_x[i] = currRow

########################################

  valid_batch_len = len(valid_data) // window_size

  valid_x_lu = []
  valid_y_lu = []
  for x in range(valid_batch_len):
    valid_x_lu.append(valid_data[x*window_size:x*window_size + window_size])
    labels_mode = \
      stats.mode(valid_labels[x*window_size:x*window_size + window_size])
    next_labels_row = [0]*(maxLabel+1)
    next_labels_row[labels_mode[0][0]] = 1
    valid_y_lu.append(next_labels_row)

########################################

  test_batch_len = len(test_data) // window_size

  test_x_lu = []
  test_y_lu = []
  test_y_expanded = []
  for x in range(test_batch_len):
    test_x_lu.append(test_data[x*window_size:x*window_size + window_size])
    labels_mode = \
      stats.mode(test_labels[x*window_size:x*window_size + window_size])
    next_labels_row = [0]*(maxLabel+1)
    next_labels_row[labels_mode[0][0]] = 1
    test_y_lu.append(next_labels_row)
    test_y_expanded.append(labels_mode[0][0])

  test_x = np.zeros(shape=(len(test_x_lu), len(embedding[0])))
  test_y = np.array(test_y_lu)


  for i in range(len(test_x_lu)):
    currRow = np.zeros(shape=(len(embedding[0]), ))
    for j in range(len(test_x_lu[0])):
      nextEmb = np.array(embedding[test_x_lu[i][j]])
      # print nextEmb.shape, currRow.shape
      currRow += nextEmb

    test_x[i] = currRow

  return train_x_lu, train_y_lu, valid_x_lu, valid_y_lu, test_x_lu, test_y_lu #, test_y_expanded

# def main():
#   train_x, train_y, test_x, test_y, test_y_expanded = makeData()
#
#   print "Partitioned size of data for training"
#   print train_x.shape
#   print train_y.shape
#
#   print "Partitioned size of data for testing"
#   print test_x.shape
#   print test_y.shape
#
#   x = tf.placeholder("float", shape=[None, train_x.shape[1]])
#   W = tf.Variable(tf.zeros([train_x.shape[1],10]))
#   b = tf.Variable(tf.zeros([10]))
#   psm = tf.matmul(tf.nn.l2_normalize(x, 0),W)
#   ps = psm + b
#   y = tf.nn.softmax(ps)
#   y_ = tf.placeholder("float", shape=[None, 10])
#   cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
#   train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#   pred = tf.argmax(y,1)
#   correct_prediction = tf.equal(pred, tf.argmax(y_,1))
#   accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
#   # train data and get results for batches
#   init = tf.initialize_all_variables()
#   sess = tf.Session()
#   sess.run(init)
#
#   lastOne = []
#
#   # train the data
#   for i in range(numIter):
#       sess.run(train_step, feed_dict={x: train_x, y_: train_y})
#       print('Training Step:' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={x: test_x, y_: test_y})) + '  Loss = ' + str(sess.run(cross_entropy, {x: train_x, y_: train_y})))
#       if i == numIter-1:
#         lastOne = sess.run(pred, feed_dict={x: test_x, y_: test_y})
#
#
#   test_y_expanded = np.array(test_y_expanded)
#   lastOne = np.array(lastOne)
#
#   # Compute confusion matrix
#   cnf_matrix = confusion_matrix(test_y_expanded, lastOne)
#   np.set_printoptions(precision=2)
#
#   # Plot non-normalized confusion matrix
#   plt.figure()
#   # class_names = ['Charles Darwin', 'Edgar Allan Poe', 'Edward Stratemeyer',\
#   #     'Jacob Abbott', 'Lewis Carroll','Mark Twain',\
#   #      'Michael Faraday', 'Ralph Waldo Emerson', \
#   #      'Rudyard Kipling', 'Winston Churchill']
#   class_names = ['CD', 'EAP', 'ES', 'JA', 'LC','MT', \
#                  'MF', 'RWE', 'RK', 'WC']
#   plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                   title='Baseline Softmax Confusion Matrix')
#
#   plt.show()
#
# if __name__ == '__main__':
#   main()
