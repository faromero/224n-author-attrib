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
import itertools
import os

def _file_to_word_ids(filename, word_to_id):
 	data = _read_words_all(filename)
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

def _read_words_all(filename):
 	words = []
 	with open(filename) as text:
 		for line in text:
 			lineWords = line.split()
 			for lw in lineWords:
 				words.append(lw)
 	return words

def _build_vocab(directory):
	data = set()
	for filename in os.listdir(directory):
		data.update(_read_words(directory + "/"  + filename))

	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))
	return word_to_id


def classifier(X, y, model, embedding_matrix, summing):
	if not summing:
		X_Glove = np.zeros(shape=(len(X), len(embedding_matrix[0])*len(X[0])))
		for i in range(len(X)):
			glove_vec = None
			for j in range(len(X[0])):
				if glove_vec is None:
					glove_vec = embedding_matrix[X[i][j]]
				else:
					glove_vec = np.append(glove_vec, embedding_matrix[X[i][j]])
			X_Glove[i] = glove_vec
	else:
		X_Glove = np.zeros(shape=(len(X), len(embedding_matrix[0])))
		for i in range(len(X)):
			glove_vec = np.zeros(len(embedding_matrix[0]))
			for j in range(len(X[0])):
				glove_vec += embedding_matrix[X[i][j]]
			X_Glove[i] = glove_vec

	X_train, X_test, y_train, y_test = train_test_split(X_Glove, y, test_size=0.1)

	y_train = np.array(y_train)
	y_test = np.array(y_test)

	# # print "XTRAIN SHAPE"
	# print X_train.shape
	# print ""
	# print "XTEST SHAPE"
	# print X_test.shape
	# print ""
	# print "YTRAIN SHAPE"
	# print y_train.shape
	# print ""
	# print "YTEST SHAPE"
	# print y_test.shape
	# print ""


	#X_train, X_dev, y_train, y_dev = train_test_split(X_train1, y_train1, test_size=0.125, random_state=42)

	if model is "NB":
		clf = MultinomialNB(fit_prior=False)
	if model is "GDA":
		clf = LinearDiscriminantAnalysis()
	if model is "SVM":
		clf = svm.SVC()
	clf.fit(X_train, y_train)
	y_hat_train = clf.predict(X_train)
	#y_hat_dev = clf.predict(X_dev)
	y_hat_test = clf.predict(X_test)
	return (y_train, y_hat_train), (y_test, y_hat_test)
	#return (y_train, y_hat_train), (y_dev, y_hat_dev), (y_test, y_hat_test)

def getAccuracy(dset):

	return np.mean([1 if dset[0][i] == dset[1][i] else 0 for i in range(len(dset[0]))])

def printResultsAndConfusionMatrix(train, dev, test, testing):
	target_names = ['Charles Darwin', 'Edgar Allan Poe', 'Edward Stratemeyer',\
					'Jacob Abbott', 'Lewis Carroll','Mark Twain',\
					 'Michael Faraday', 'Ralph Waldo Emerson', \
					 'Rudyard Kipling', 'Winston Churchill']
	print "TRAINING RESULTS"
	print(classification_report(train[0], train[1], target_names=target_names))
	print""
	print""
	print""
	if testing:
		print "TEST RESULTS"
		print (classification_report(test[0], test[1], target_names=target_names))
	else:
		print "DEV RESULTS"
		print(classification_report(dev[0], dev[1], target_names=target_names))
		print""
		print""

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

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

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




if __name__ == "__main__":
	testing = True
	summing = True
	numKfold = 10
	if not testing:
		
		data_directory = "clean_books_div"
		glovePath = "../glove.6B/glove.6B.50d.txt"
		model = "GDA"    #RF, Gaussian, NB
		#windowSize = 200

		# Map the words to integer IDs
		word_to_id = _build_vocab(data_directory)

		# Read in GloVe vectors and store in a numpy array
		embedding_matrix = build_embedding(word_to_id, glovePath)
		embedding_matrix = np.array(embedding_matrix)
		if model == "NB":
			embedding_matrix -= np.amin(embedding_matrix)
		for windowSize in range(10,2000,100):
			strideLength = int(windowSize)
			trainingExamples = ([],[])
			currLabel = 0

			# Convert the file to word ids
			for inputFile in os.listdir(data_directory):

				data = _file_to_word_ids(data_directory + "/" + inputFile, word_to_id)
				print len(data)
				# Append examples 
				for i in range(0, len(data) - windowSize, strideLength):
					
					trainingExamples[0].append(data[i:i + windowSize])
					trainingExamples[1].append(currLabel)
				currLabel += 1


				# see how well it does with multinomial naive bayes
			print "WINDOW SIZE =", windowSize
			#train, dev, test = classifier(trainingExamples[0], trainingExamples[1], \
											#model, embedding_matrix, summing)
			#printResultsAndConfusionMatrix(train, dev, test, testing)
			trainAcc = 0
			testAcc = 0
			for _ in range(numKfold):
				train, test = classifier(trainingExamples[0], trainingExamples[1], \
												model, embedding_matrix, summing)

				trainAcc += getAccuracy(train)
				testAcc += getAccuracy(test)

			print "Train Accuracy"
			print 1.0 * trainAcc / numKfold
			print ""
			print "Test Accuracy"
			print 1.0 * testAcc / numKfold
			print ""
			# print "i"

	else:
		wsNB = 200
		wsGDA = 1000
		modelDict = {"NB" : wsNB, "GDA" : wsGDA}
		modelDict= {"GDA" : wsGDA}
		for mod in modelDict:
			data_directory = "clean_books_div"
			glovePath = "../glove.6B/glove.6B.300d.txt"
			model = mod   
			windowSize = modelDict[mod]
			strideLength = int(windowSize) + 10

			# Map the words to integer IDs
			word_to_id = _build_vocab(data_directory)

			# Read in GloVe vectors and store in a numpy array
			embedding_matrix = build_embedding(word_to_id, glovePath)
			embedding_matrix = np.array(embedding_matrix)
			if model == "NB":
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

			train, test = classifier(trainingExamples[0], trainingExamples[1], \
				model, embedding_matrix, summing)
			#printResultsAndConfusionMatrix(train, dev, test, testing)
	
			# Compute confusion matrix
			cnf_matrix = confusion_matrix(test[0], test[1])
			np.set_printoptions(precision=2)

			# Plot non-normalized confusion matrix
			plt.figure()
			class_names = ['Charles Darwin', 'Edgar Allan Poe', 'Edward Stratemeyer',\
					'Jacob Abbott', 'Lewis Carroll','Mark Twain',\
					 'Michael Faraday', 'Ralph Waldo Emerson', \
					 'Rudyard Kipling', 'Winston Churchill']
			plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                      title='normalized')

			
			plt.show()









