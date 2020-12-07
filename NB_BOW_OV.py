import csv
from itertools import chain
import math
from nltk.tokenize import RegexpTokenizer

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


# To remove duplicates from the vocabulary
vocabulary = list(set(vocabulary))
seen = set()
finalVocabulary = []
for item in vocabulary:
    if item not in seen:
        seen.add(item)
        finalVocabulary.append(item)


# COMPUTE CONDITIONALS
# Vocabulary = all words in the tweets of the training set

# Step 1: Split the tweets based on yes or no

numTweetsNo = 0
numTweetsYes = 0
yesTweets = []
noTweets = []
finalYesTweets = []
finalNoTweets = []

for i in range(numrows):
    if arr[i][2] == "no":
        numTweetsNo += 1
        noTweets.append(arr[i][1])
    if arr[i][2] == "yes":
        numTweetsYes += 1
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


# COMPUTE PRIORS
# Calculate probabilities of each class (yes, no for factual tweet)
totalTweets = numTweetsNo + numTweetsYes
priorYes = numTweetsYes / totalTweets
priorNo = numTweetsNo / totalTweets

# Apply the model on the test set

tsv_file = open("covid_test_public.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")
testArr = []
for row in read_tsv:
    testArr.append(row)


testTweets = []
for i in range(len(testArr)):
    tokenizer = RegexpTokenizer(r'\w+')
    s = tokenizer.tokenize(testArr[i][1])
    testTweets.append(s)


def calculateCondYes(freq):
    prob = math.log10((freq + 0.01) / (totalWordsYes + len(finalVocabulary)))
    return prob


def calculateCondNo(freq):
    prob = math.log10((freq + 0.01) / (totalWordsNo + len(finalVocabulary)))
    return prob

# Total Yes conditionals
scoreYes = 1  # can't be 0 to start
for item in testTweets[2]:
    if item in yesDictionary:
        scoreYes *= calculateCondYes(yesDictionary[item])

scoreYes *= math.log10(priorYes)
print("Score Yes : " + str(scoreYes))

# Total No conditionals
scoreNo = 1  # can't be 0 to start
for item in testTweets[2]:
    if item in noDictionary:
        scoreNo *= calculateCondNo(noDictionary[item])

scoreNo *= math.log10(priorNo)
print("Score No : " + str(scoreNo))

# Writing to the Trace file
f= open("trace_NB-BOW-OV.txt","w+")
for i in range(len(testArr)):
     f.write(testArr[i][0]+"  yes/no  classScore  "+testArr[i][2]+"  correct\n")

# Writing to the Evaluation file
f= open("eval_NB-BOW-OV.txt","w+")
f.write("Accuracy\n"+"yes-Precision  no-Precision\n"+"yes-recall  no-recall\n"+"yes-F1  no-F1")

f.close()


