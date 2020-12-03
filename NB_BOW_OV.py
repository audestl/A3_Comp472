import csv

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

numrows = len(arr)
numcols = len(arr[0])

# ClEANING THE DATA AND PREPARE IT FOR VOCABULARY

# Fold training set in lowercase
for i in range(numrows):
    arr[i][1] = arr[i][1].lower()



# # Build a list of all words in training set (bag of words)
#
#
# tokenizer = RegexpTokenizer(r'\w+')
# s = tokenizer.tokenize(arr[5][1])
#
# print(s)

s = arr[7][1]
t = re.findall(r'\S+', s)
print(t)

# COMPUTE CONDITIONALS
# Vocabulary = all words in the tweets of the training dataset

# Step 1: Split the tweets based on yes or no

yesTweets = []
noTweets = []

for i in range(numrows):
    if arr[i][2] == "yes":
        yesTweets.append(arr[i].copy())
    if arr[i][2] == "no":
        noTweets.append(arr[i].copy())



# COMPUTE PRIORS
# Calculate probabilities of each class (yes, no for factual tweet)
numNo = 0
numYes = 0

for i in range(numrows):
    if arr[i][2] == "no":
        numNo += 1
    if arr[i][2] == "yes":
        numYes += 1

probYes = numYes / (numYes + numNo)
probNo = numNo / (numYes + numNo)
