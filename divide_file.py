#! /usr/bin/python

import glob
import sys
import numpy as np

totFileDir = 'clean_books_div/'
trainSize = 0.8
devSize = 0.1
# testSize = 0.2, not needed

authorLabelDict = {
  "Charles Darwin": 0,
  "Edgar Allan Poe": 1,
  "Edward Stratemeyer": 2,
  "Jacob Abbott": 3,
  "Lewis Carroll": 4,
  "Mark Twain": 5,
  "Michael Faraday": 6,
  "Ralph Waldo Emerson": 7,
  "Rudyard Kipling": 8,
  "Winston Churchill": 9  
}

fidTrain = open(totFileDir + 'guten_train.txt', 'w')
fidDev = open(totFileDir + 'guten_dev.txt', 'w')
fidTest = open(totFileDir + 'guten_test.txt', 'w')

for currBook in glob.glob(totFileDir + '*_all.txt'):  
  currBookName = currBook.split('/')[1].split('_')[0]
  label = str(authorLabelDict[currBookName])

  fidOrig = open(currBook, 'r')
  allLines = fidOrig.readlines()
  numLines = len(allLines)
  trainCutoff = np.floor(numLines * trainSize)
  devCutoff = np.floor(numLines * (trainSize + devSize))
  
  print currBook, 'has', numLines, 'lines.'
  
  for i, line in enumerate(allLines):
    lineSplit = line.split()
    for x in range(len(lineSplit)):
      lineSplit[x] = label+lineSplit[x]
    
    line2Write = ' '.join(lineSplit)

    if i < trainCutoff:
      fidTrain.write(line2Write)
    elif i < devCutoff:
      fidDev.write(line2Write)
    else:
      fidTest.write(line2Write)
  
  fidOrig.close()

fidTrain.close()
fidDev.close()
fidTest.close()
