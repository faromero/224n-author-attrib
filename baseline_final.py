# Baseline for cs224n project
# Written by Gregory Luppescu and Francisco Romero

import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from scipy import stats
from math import factorial
import itertools
import os

def build_embedding(word_to_id, glovePath):
  dimSize = glovePath.split('.')[-2]
  dimSize = int(dimSize.strip('d'))
  embedding_matrix = np.random.uniform(size=(len(word_to_id), dimSize), \
    low=-1.0, high=1.0)

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

def _file_to_word_ids(filename, word_to_id):
	data = _read_words(filename, omit=False)
	x = [word_to_id[word[1:]]  for word in data if word[1:] in word_to_id]
	y = [int(word[0])  for word in data if word[1:] in word_to_id]
	return (x,y)


def _read_words(filename, omit):
	words = []
	with open(filename) as text:
		for line in text:
			lineWords = line.split()
			for lw in lineWords:
				if omit:
					words.append(lw[1:])
				else:
					words.append(lw)
	return words

def _build_vocab_train_dev(trainFile, devFile):
	data = set()
	fnames = [trainFile, devFile]
	for filename in fnames:
		data.update(_read_words(filename, omit=True))

	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))
	return word_to_id

def _build_vocab(filename):
	data = _read_words(filename, omit=True)
	data = set(data)
	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))
	
	return word_to_id

def inds_2_GloVe(X, embedding_matrix):
	X_Glove = np.zeros(shape=(len(X), len(embedding_matrix[0])))
	for i in range(len(X)):
		glove_vec = np.zeros(len(embedding_matrix[0]))
		for j in range(len(X[0])):
			glove_vec += embedding_matrix[X[i][j]]
		X_Glove[i] = glove_vec
	
	return X_Glove

def plot_confusion_matrix(cm, classes,
                          normalize=False,
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


def getAccuracy(dset):
	return np.mean([1 if dset[0][i] == dset[1][i] else 0 for i in range(len(dset[0]))])

def classifier_train_dev(trainingExamples, devExamples, embedding_matrix_train_dev, model):
	X_train = trainingExamples[0]
	y_train = trainingExamples[1]
	X_dev= devExamples[0]
	y_dev = devExamples[1]

	train_GloVe = inds_2_GloVe(X_train, embedding_matrix_train_dev)
	dev_GloVe = inds_2_GloVe(X_dev, embedding_matrix_train_dev)

	if model == "NB":
		clf = MultinomialNB(fit_prior=True)
	if model == "GDA":
		clf = LinearDiscriminantAnalysis()
	if model == "SVM":
		clf = svm.SVC(decision_function_shape='ovo')

	clf.fit(train_GloVe, y_train)

	y_hat_train = clf.predict(train_GloVe)
	y_hat_dev = clf.predict(dev_GloVe)

	return (y_train, y_hat_train), (y_dev, y_hat_dev) 



def classifier(trainingExamples, devOrTestExamples, embedding_matrix_train,
			   embedding_matrix_DOrT, model):
	X_train = trainingExamples[0]
	y_train = trainingExamples[1]
	X_DOrT = devOrTestExamples[0]
	y_DorT = devOrTestExamples[1]

	train_GloVe = inds_2_GloVe(X_train, embedding_matrix_train)
	dev_GloVe = inds_2_GloVe(X_DOrT, embedding_matrix_DOrT)

	if model == "NB":
		clf = MultinomialNB(fit_prior=True)
	if model == "GDA":
		clf = LinearDiscriminantAnalysis()
	if model == "SVM":
		clf = svm.SVC(decision_function_shape='ovo')

	clf.fit(train_GloVe, y_train)

	y_hat_train = clf.predict(train_GloVe)
	y_hat_DOrT = clf.predict(dev_GloVe)

	return (y_train, y_hat_train), (y_DorT, y_hat_DOrT) 

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


if __name__ == "__main__":

	# some constants
	doingDev = False
	trainFilename = "../guten_train.txt"
	devFilename = "../guten_dev.txt"
	testFilename = "../guten_test.txt"
	gloveFilename = "../glove.42B.300d.txt"
	windowBeg = 10
	windowEnd = 1501
	increment = 10
	kFold = 1
	model = "GDA"  #GDA or NB

	# Build Embeddings and data sets for train, dev, and test **********************************

	#Original train
	word_to_id_train_dev = _build_vocab_train_dev(trainFilename, devFilename)
	embedding_train_dev = np.array(build_embedding(word_to_id_train_dev, gloveFilename))
	data_train = _file_to_word_ids(trainFilename, word_to_id_train_dev)
	data_dev = _file_to_word_ids(devFilename, word_to_id_train_dev)

	#test
	word_to_id_test = _build_vocab(testFilename)
	embedding_test = np.array(build_embedding(word_to_id_test, gloveFilename))
	data_test = _file_to_word_ids(testFilename, word_to_id_test)

	print "READ IN EVERYTHING"
	if model == "NB":
		embedding_train_dev -= np.amin(embedding_train_dev)
		embedding_test -= np.amin(embedding_test)

	if doingDev:

		# For each window size, run learning over training and dev set *****************************
		trainAccs = []
		devAccs = []
		for windowSize in range(windowBeg, windowEnd, increment):
			strideLength = windowSize
			
			# form training examples
			trainingExamples = ([],[])
			for i in range(0, len(data_train[0]) - windowSize, strideLength):
				trainingExamples[0].append(data_train[0][i:i + windowSize])
				label = stats.mode(data_train[1][i:i + windowSize])[0][0]
				trainingExamples[1].append(label)

			# form dev examples
			devExamples = ([],[])
			for i in range(0, len(data_dev[0]) - windowSize, strideLength):
				devExamples[0].append(data_dev[0][i:i + windowSize])
				label = stats.mode(data_dev[1][i:i + windowSize])[0][0]
				devExamples[1].append(label)

			devLen = len(devExamples[0])

			data_combinedX = []
			data_combinedX.extend(devExamples[0])
			data_combinedX.extend(trainingExamples[0])


			data_combinedY = []
			data_combinedY.extend(devExamples[1])
			data_combinedY.extend(trainingExamples[1])
			data_combined = (data_combinedX, data_combinedY) 

			print "SIZE OF TRAIN: ", len(data_train[0]) 
			print "SIZE OF DEV: ", len(data_dev[0])
			print "SIZE OF COMBINED: ", len(data_combined[0])

			trainAcc = 0
			devAcc = 0
			for _ in range(kFold):
				
				shuffledX, shuffledY = unison_shuffled_copies(np.array(data_combined[0]), np.array(data_combined[1]))
				devExamples0 = shuffledX[:devLen]
				devExamples1 = shuffledY[:devLen]
				trainingExamples0 = shuffledX[devLen:]
				trainingExamples1 = shuffledY[devLen:]
				trainingExamples = (trainingExamples0, trainingExamples1)
				devExamples = (devExamples0, devExamples1)

				print "SIZE OF SHUFFLEDX", len(shuffledX)
				print "SIZE of SHUFFLEDY", len(shuffledY)


				print "DEV EXAMPLES****************************"
				print devExamples

				print "TRAIN EXAMPLES*********************"
				print trainingExamples
				# Train model and evaluate dev set
				train, dev = classifier_train_dev(trainingExamples, devExamples, \
										embedding_train_dev, model)

				trainAcc += getAccuracy(train)
				devAcc += getAccuracy(dev)

			print "WINDOW SIZE: ", windowSize
			print "TRAIN ACCURACY: ", (1.0 * trainAcc) / kFold
			print "DEV ACCURACY: ", (1.0 * devAcc) / kFold
			print ""
			print ""

			trainAccs.append((1.0 * trainAcc) / kFold)
			devAccs.append((1.0 * devAcc) / kFold)

		if model == "GDA":
			np.save('trainAccs.npy', trainAccs)
			np.save('devAccs.npy', devAccs)

		if model == "NB":
			np.save('trainAccs_NB.npy', trainAccs)
			np.save('devAccs_NB.npy', devAccs)

		if model == "SVM":
			np.save('trainAccs_SVM.npy', trainAccs)
			np.save('devAccs_SVM.npy', devAccs)
	else:
		if model == "GDA":
			windowSize = np.load('GDA_window_opt.npy')

		if model == "NB":
			windowSize= np.load('NB_window_opt.npy')

		strideLength = windowSize

		trainingExamples = ([],[])
		for i in range(0, len(data_train[0]) - windowSize, strideLength):
			trainingExamples[0].append(data_train[0][i:i + windowSize])
			label = stats.mode(data_train[1][i:i + windowSize])[0][0]
			trainingExamples[1].append(label)

		# form dev examples
		testExamples = ([],[])
		for i in range(0, len(data_test[0]) - windowSize, strideLength):
			testExamples[0].append(data_test[0][i:i + windowSize])
			label = stats.mode(data_test[1][i:i + windowSize])[0][0]
			testExamples[1].append(label)


		# Train model and evaluate dev set
		train, test = classifier(trainingExamples, testExamples, \
								embedding_train_dev, embedding_test, model)

		if model == "GDA":
			np.save("testGDA", test)
		if model == "NB":
			np.save("testNB", test)

		trainAcc = getAccuracy(train)
		testAcc = getAccuracy(test)

		print "Model: ", model
		print "WINDOW SIZE: ", windowSize
		print "TRAIN ACCURACY: ", trainAcc
		print "TEST ACCURACY: ", testAcc

		cnf_matrix = confusion_matrix(test[0], test[1])
		np.set_printoptions(precision=2)

		print cnf_matrix
		# Plot non-normalized confusion matrix
		plt.figure()
		class_names = ['CD', 'EAP', 'ES',\
				'JA', 'LC', 'MT',\
				 'MF', 'RWE', \
				 'RK', 'WC']

		if model == "NB":
			plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                  title='Baseline Multinomial Naive Bayes Confusion Matrix')
		if model == "GDA":
			plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                  title='Baseline Gaussian Discriminant Analysis Confusion Matrix')


		
		plt.show()

	
	


