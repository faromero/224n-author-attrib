#! /usr/bin/python

import os
import sys
import nltk
import glob
import argparse
import numpy as np

stupidNames = ['in', 'an', 'many', 'love', 'precious', 'king', 'long', \
              'forest', 'sage', 'chance', 'ai', 'sun', 'rose', 'golden', \
              'so', 'miles', 'my', 'son', 'see', 'may', 'else', 'spring', \
              'soon', 'young', 'miss', 'man', 'moon', 'van', 'marry', \
              'autumn', 'summer', 'song', 'season', 'sang']

rawBookDir = 'test/'
cleanBookDir = 'clean_books_test/'
gatherBookDir = 'clean_books_div/'
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
    print 'Getting counts for:', currBook
    fid = open(currBook, 'r')
    rawBook = fid.read().decode('utf-8')
    wordTokens = nltk.word_tokenize(rawBook)
    lowerWordTokens = [wl.lower() for wl in wordTokens]
    lowerWordTokens = \
      [t.replace("``", '"').replace("''", '"') for t in lowerWordTokens]

    for wl in range(len(lowerWordTokens)):
      lowerWordTokens[wl] = lowerWordTokens[wl].\
        rstrip('-').strip('-').rstrip('_').strip('_')      

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
    rawBook = fidRaw.read().decode('utf-8')
    sentTokens = nltk.sent_tokenize(rawBook)

    # For each sentence, tokenize and make all words lowercase
    for sent in sentTokens:
      wordTokens = nltk.word_tokenize(sent)
      lowerWordTokens = [wl.lower() for wl in wordTokens]
      lowerWordTokens = \
        [t.replace("``", '"').replace("''", '"') for t in lowerWordTokens]
      for wl in range(len(lowerWordTokens)):
        lowerWordTokens[wl] = lowerWordTokens[wl].\
          rstrip('-').strip('-').rstrip('_').strip('_')

      for w in range(len(lowerWordTokens)):
        if lowerWordTokens[w] in namesDB:
          lowerWordTokens[w] = '<name>'
        elif lowerWordTokens[w] in unkWordSet:
          lowerWordTokens[w] = '<unk>'

      # Join words
      cleanSent = ' '.join(lowerWordTokens)
      fidClean.write(cleanSent.encode('utf-8') + '\n')

    fidClean.close()
    fidRaw.close()

  print 'All books have been cleaned! Cleaned books available in', cleanBookDir

def gatherBooks(args):

  # Check if directory exists. If not, make it.
  if not os.path.exists(gatherBookDir):
    os.makedirs(gatherBookDir)

  lineCap = args.linesCap
  uniqueAuthorSet = set()

  for currBook in glob.glob(cleanBookDir + '*.txt'):
    currbookSplit = currBook.split('___')[0].split('/')
    uniqueAuthorSet.add(currbookSplit[1])


  for author in uniqueAuthorSet:
    fidAll = open(gatherBookDir + author + '_all.txt', 'w')
    currLines = 0
    for currBook in glob.glob(cleanBookDir + author + '*.txt'):
      fidRead = open(currBook, 'r')

      for line in fidRead:
        if currLines < lineCap:
          fidAll.write(line)
          currLines += 1
        else:
          break

      if currLines >= lineCap:
        break

      fidRead.close()
    fidAll.close()

  print 'Concatenation complete, now getting line counts'
  for currBook in glob.glob(gatherBookDir + '*_all.txt'):
    fid = open(currBook, 'r')
    allLines = fid.readlines()
    bookTitle = currBook.split('/')[1]
    print bookTitle, 'lines:', len(allLines)

  print 'Line counting complete!'

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('--th', type=float, default=0.0001,
                  help='Perc of total tokens to keep (default: 0.0001)')
  ap.add_argument('--linesCap', type=int, default=12000,
                  help='Line limit per author (default: 12000)')
  args = ap.parse_args()
  
  unkWords = countAndCalculate(args)
  tokenizeAndClean(unkWords)
  gatherBooks(args)
  