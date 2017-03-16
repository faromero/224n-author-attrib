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
from sklearn.ensemble import RandomForestClassifier
import os

def _file_to_word_ids(filename, word_to_id):
 	data = _read_words(filename)
 	return [word_to_id[word] for word in data if word in word_to_id]


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

def _read_words(filename):
	words = []
  	with open(filename) as text:
  		for line in text:
  			lineWords = line.split()
  			for lw in lineWords:
  				words.append(lw)
  	return set(words)

def _build_vocab(directory):
	data = set()
	for filename in os.listdir(directory):
		data.update(_read_words(directory + "/"  + filename))

	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))
	return word_to_id


def classifier(X, y, model, embedding_matrix):
	print len(X)
	X_Glove = np.zeros(shape=(len(X), len(embedding_matrix[0])*len(X[0])))
	for i in range(len(X)):
		glove_vec = None
		for j in range(len(X[0])):
			if glove_vec is None:
				glove_vec = embedding_matrix[X[i][j]]
			else:
				glove_vec = np.append(glove_vec, embedding_matrix[X[i][j]])
		X_Glove[i] = glove_vec
	print "DONE!"

	X_train, X_test, y_train, y_test = train_test_split(X_Glove, y, test_size=0.1)
	X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1)

	if model is "NB":
		clf = MultinomialNB(fit_prior=False, alpha=0)
	if model is "Gaussian":
		clf = LinearDiscriminantAnalysis()
	if model is "RF":
		clf = RandomForestClassifier(n_estimators=3)
	clf.fit(X_train, y_train)
	y_hat_train = clf.predict(X_train)
	y_hat_dev = clf.predict(X_dev)
	y_hat_test = clf.predict(X_test)
	return (y_train, y_hat_train), (y_dev, y_hat_dev), (y_test, y_hat_test)

def printResultsAndConfusionMatrix(train, dev, test):
	target_names = ['Book 1', 'Book 2', 'Book 3', 'Book 4', 'Book 5',\
					'Book 6', 'Book 7', 'Book 8', 'Book 9', 'Book10']
	print "TRAINING RESULTS"
	print(classification_report(train[0], train[1], target_names=target_names))
	print""
	print""
	#print confusion_matrix(train[0], train[1])
	print""
	print "DEV RESULTS"
	print(classification_report(dev[0], dev[1], target_names=target_names))
	print""
	print""
	#print confusion_matrix(test[0], test[1])




if __name__ == "__main__":
	for windowSize in range(10,100,10):
		data_directory = "clean_books_div"
		glovePath = "../glove.6B/glove.6B.300d.txt"
		model = "NB"    #RF, Gaussian, NB
		#windowSize = 200
		strideLength = int(windowSize)

		# Map the words to integer IDs
		word_to_id = _build_vocab(data_directory)

		# Read in GloVe vectors and store in a numpy array
		embedding_matrix = build_embedding(word_to_id, glovePath)
		embedding_matrix = np.array(embedding_matrix)
		embedding_matrix -= np.amin(embedding_matrix)

		trainingExamples = ([],[])
		currLabel = 0

		# Convert the file to word ids
		for inputFile in os.listdir(data_directory):

			data = _file_to_word_ids(data_directory + "/" + inputFile, word_to_id)
				
			# Append examples 
			for i in range(0, len(data) - windowSize, strideLength):
				trainingExamples[0].append(data[i:i + windowSize])
				trainingExamples[1].append(currLabel)

			currLabel += 1		

		# see how well it does with multinomial naive bayes
		print "WINDOW SIZE =", windowSize
		train, dev, test = classifier(trainingExamples[0], trainingExamples[1], model, embedding_matrix)

		printResultsAndConfusionMatrix(train, dev, test)





