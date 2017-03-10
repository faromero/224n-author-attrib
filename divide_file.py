#! /usr/bin/python

import glob
import numpy as np
# 705925
totFileDir = 'clean_books_div/'
trainSize = 0.8
devSize = 0.1
# testSize = 0.2, not needed

fidTrain = open(totFileDir + 'guten_train.txt', 'w')
fidDev = open(totFileDir + 'guten_dev.txt', 'w')
fidTest = open(totFileDir + 'guten_test.txt', 'w')

for currBook in glob.glob(totFileDir + '*_all.txt'):  
  currBookName = currBook.split('.')[0]

  fidOrig = open(currBook, 'r')
  allLines = fidOrig.readlines()
  numLines = len(allLines)
  trainCutoff = np.floor(numLines * trainSize)
  devCutoff = np.floor(numLines * (trainSize + devSize))
  
  print currBook, 'has', numLines, 'lines.'
  
  for i, line in enumerate(allLines):
    if i < trainCutoff:
      fidTrain.write(line)
    elif i < devCutoff:
      fidDev.write(line)
    else:
      fidTest.write(line)
  
  fidOrig.close()

fidTrain.close()
fidDev.close()
fidTest.close()


# cat clean_books_test/Sir\ Arthur\ Conan\ Doyle___* > clean_books_div/SACD_all.txt; cat clean_books_test/Walt\ Whitman___* > clean_books_div/WW_all.txt; cat clean_books_test/Mark\ Twain___* > clean_books_div/MT_all.txt; cat clean_books_test/Herman\ Melville___* > clean_books_div/HM_all.txt; cat clean_books_test/Charles\ Dickens___* > clean_books_div/CD_all.txt;
