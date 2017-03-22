import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from math import factorial
import itertools
import os

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

model = "DEEP"

if model == "GDA":
	trainAccs = np.load("trainAccs.npy")
	devAccs = np.load("devAccs.npy")

if model == "NB":
	trainAccs = np.load("trainAccs_NB.npy")
	devAccs = np.load("devAccs_NB.npy")

if model == "SVM":
	trainAccs = np.load("trainAccs_SVM.npy")
	devAccs = np.load("trainAccs_SVM.npy")

if model == "DEEP":
	trainAccs = []
	devAccs = []
	filename = "acc_plt.txt"
	with open(filename) as text:
		for line in text:
			accs = line.split()
			if float(accs[1]) > .05:
				trainAccs.append(accs[0])
				devAccs.append(accs[1])
			else:
				trainAccs.append(trainAccs[len(trainAccs) - 1])
				devAccs.append(devAccs[len(devAccs) - 1])

#devAccs = savitzky_golay(devAccs, 21, 11)
#windowSizes = np.arange(10, 1001, 10)
epochs = np.arange(1,301,1)
# indDmax = np.argmax(devAccs[:len(windowSizes)])

# print "BEST WINDOW SIZE", windowSizes[indDmax]
# print "BEST TRAINING", trainAccs[indDmax]
# print "BEST DEV", devAccs[indDmax]

plt.plot(epochs, trainAccs, label='Train Accuracy')
plt.plot(epochs, devAccs, label='Dev Accuracy')
#plt.axvline(windowSizes[indDmax], color='r', linestyle='--', label='Optimal Window Size')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
if model == "NB":
	plt.title('Accuracy vs. Window Size Using Multinomial Naive Bayes')
	np.save("NB_window_opt.npy", windowSizes[indDmax])
if model == "GDA":
	plt.title('Accuracy vs. Window Size Using Gaussian Discriminant Analysis')
	np.save("GDA_window_opt.npy", windowSizes[indDmax])
if model == "DEEP":
	plt.title('Accuracy vs. Epoch for LSTM Model')
plt.legend(bbox_to_anchor=(0., 1.02, .84, .102), loc=1,
           ncol=3, borderaxespad=0)
plt.show()

