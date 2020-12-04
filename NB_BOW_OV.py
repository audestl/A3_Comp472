import csv
from itertools import chain

import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import re

tsv_file = open("covid_training.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")
arr = []

for row in read_tsv:
    arr.append(row)

arr.remove(arr[0])
numrows = len(arr)
numcols = len(arr[0])

# ClEANING THE DATA AND PREPARE IT FOR VOCABULARY

# Fold training set in lowercase
for i in range(numrows):
    arr[i][1] = arr[i][1].lower()

vocabularyArr = []
vocabulary = []
# # Build a list of all words in training set (bag of words)
#
#
for i in range(numrows):
    tokenizer = RegexpTokenizer(r'\w+')
    s = tokenizer.tokenize(arr[i][1])
    vocabularyArr.append(s)

for i in range(numrows):
    singleSize = len(vocabularyArr[i])
    for y in range(singleSize):
        vocabulary.append(vocabularyArr[i][y])

# vocabulary.remove(vocabulary[0])
# vocabulary = list(chain.from_iterable(vocabulary))

# To remove duplicates from the vocabulary
vocabulary = list(set(vocabulary))
seen = set()
finalVocabulary = []
for item in vocabulary:
    if item not in seen:
        seen.add(item)
        finalVocabulary.append(item)

# ANOTHER WAY TO CLEAN THE DATA
# s = arr[7][1]
# t = re.findall(r'\S+', s)
# print(t)

# COMPUTE CONDITIONALS
# Vocabulary = all words in the tweets of the training dataset

# Step 1: Split the tweets based on yes or no

numNo = 0
numYes = 0
yesTweets = []
noTweets = []
finalYesTweets = []
finalNoTweets = []

for i in range(numrows):
    if arr[i][2] == "no":
        numNo += 1
        noTweets.append(arr[i][1])
    if arr[i][2] == "yes":
        numYes += 1
        yesTweets.append(arr[i][1])

# Step 2 : Count each instances of every word from the vocabulary in YesTweets
sizeYes = len(yesTweets)

for i in range(sizeYes):
    tokenizer = RegexpTokenizer(r'\w+')
    s = tokenizer.tokenize(yesTweets[i])
    finalYesTweets.append(s)
finalYesTweets = list(chain.from_iterable(finalYesTweets))

# Step 3 : Count each instances of every word from the vocabulary in NoTweets

sizeNo = len(noTweets)
for i in range(sizeNo):
    tokenizer = RegexpTokenizer(r'\w+')
    s = tokenizer.tokenize(noTweets[i])
    finalNoTweets.append(s)
finalNoTweets = list(chain.from_iterable(finalNoTweets))

# Count number of words in each dictionary
totalWordsNo = len(finalNoTweets)
totalWordsYes = len(finalYesTweets)

yesDictionary = dict()
noDictionary = dict()
# Initialize "yes" tweets dictionary with training set
for i in range(len(finalVocabulary)):
    val = finalYesTweets.count(finalVocabulary[i])
    yesDictionary[finalVocabulary[i]] = val

# Initialize "No" tweets dictionary with training set
for i in range(len(finalVocabulary)):
    val = finalNoTweets.count(finalVocabulary[i])
    noDictionary[finalVocabulary[i]] = val


# Initialize "no" tweets dictionary with training set
for i in range(len(finalVocabulary)):
    val = finalNoTweets.count(finalVocabulary[i])
    noDictionary[finalVocabulary[i]] = val


# COMPUTE PRIORS
# Calculate probabilities of each class (yes, no for factual tweet)
probYes = numYes / (numYes + numNo)
probNo = numNo / (numYes + numNo)

# Apply the model on the test set
def computeResult():
    # Score(No) = Prior(No) x P(word1 | No) X (word2 | No) ... P (wordX | No)
    # Score(Yes) = Prior(Yes) x P(word1 | Yes) X (word2 | Yes) ... P (wordX | Yes)
    # return argmax (Score(yes), Score(no))



