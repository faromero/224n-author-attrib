#! /usr/bin/python

import os
import sys
import nltk
import glob
import argparse
import numpy as np

# will
stupidNames = ['in', 'an', 'many', 'love', 'precious', 'king', 'long', \
              'forest', 'sage', 'chance', 'ai', 'sun', 'rose', 'golden', \
              'so', 'miles', 'my', 'son', 'see', 'may', 'else', 'spring', \
              'soon', 'young', 'miss', 'man', 'moon', 'van', 'marry', \
              'autumn', 'summer', 'song', 'season', 'sang']

rawBookDir = 'gutenburg/'
cleanBookDir = 'clean_books/'
namesDBFile = 'names_db.txt'

""" Function to load in names database """
def getNamesDB():
  namesList = []

  fidNames = open(namesDBFile, 'r')

  for nameLine in fidNames:
    name = nameLine.split()[0]
    name = name.lower()
    if name not in stupidNames:
      namesList.append(name)

  fidNames.close()

  return set(namesList)

""" Function to obtain counts of tokens for determining <unk> threshold """
def countAndCalculate(args):
  keepThresh = args.th
  
  wordCounterDict = {}
  for currBook in glob.glob(rawBookDir + '*.txt'):
    fid = open(currBook, 'r')
    rawBook = fid.read()
    wordTokens = nltk.word_tokenize(rawBook)
    lowerWordTokens = [wl.lower() for wl in wordTokens]

    # Keep track of counts
    for word in lowerWordTokens:
      if word in wordCounterDict:
        wordCounterDict[word] += 1
      else:
        wordCounterDict[word] = 1

    fid.close()
  totalTokens = len(wordCounterDict)

  # Compute "keep threshold"
  cutoff = np.ceil(totalTokens * keepThresh)

  unkWords = []
  for key, value in wordCounterDict.iteritems():
    if value < cutoff:
      unkWords.append(key)

  # Print some statistics
  print 'Total number of unique tokens:', totalTokens
  print 'Cutoff count:', cutoff
  print 'Number of <unk> words:', len(unkWords)
  print

  return set(unkWords)

""" Function to tokenize and clean tokens. Should be run after
    countAndCalculate """
def tokenizeAndClean(unkWordSet):
  # Get all names
  namesDB = getNamesDB()

  # Check if directory exists. If not, make it.
  if not os.path.exists(cleanBookDir):
    os.makedirs(cleanBookDir)

  for currBook in glob.glob(rawBookDir + '*.txt'):
    bookName = currBook.split('/')[-1]
    print 'Now cleaning:', bookName

    fidClean = open(cleanBookDir + bookName, 'w')

    fidRaw = open(currBook, 'r')
    rawBook = fidRaw.read()
    sentTokens = nltk.sent_tokenize(rawBook)

    # For each sentence, tokenize and make all words lowercase
    for sent in sentTokens:
      wordTokens = nltk.word_tokenize(sent)
      lowerWordTokens = [wl.lower() for wl in wordTokens]

      # Use indexing as oppose to direct iteration to replace entries if needed
      # Need to traverse in reverse order since we may delete entries
      for w in range(len(lowerWordTokens) - 1, -1 , -1):
        if lowerWordTokens[w] in namesDB:
          # print 'Found name:', lowerWordTokens[w], ', now deleting...'
          del lowerWordTokens[w]
        elif lowerWordTokens[w] in unkWordSet:
          lowerWordTokens[w] = '<unk>'

      # Join words
      cleanSent = ' '.join(lowerWordTokens)
      fidClean.write(cleanSent + '\n')

    fidClean.close()
    fidRaw.close()

  print 'All books have been cleaned! Cleaned books available in', cleanBookDir

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('--th', type=float, default=0.001,
                  help='Percent of total number of tokens to keep')
  args = ap.parse_args()

  unkWords = countAndCalculate(args)
  tokenizeAndClean(unkWords)
  