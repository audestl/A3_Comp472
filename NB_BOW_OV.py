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

# Initialize "no" tweets dictionary with training set
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


# testTweets = list(chain.from_iterable(testTweets))

# For every word in testTweets, find it in yesDictionary
#   if not there, go to next one
#   else : pass it to condYes()
def condYes(freq):
    prob = math.log10((freq + 0.01) / (totalWordsYes + len(finalVocabulary)))
    return prob


def condNo(freq):
    prob = math.log10((freq + 0.01) / (totalWordsNo + len(finalVocabulary)))
    return prob


print(testTweets)
totalYesConditionals = 1  # can't be 0 to start
for item in testTweets[0]:
    if item in yesDictionary:
        totalYesConditionals *= condYes(yesDictionary[item])

totalYesConditionals *= math.log10(priorYes)
print("Score Yes : " + str(totalYesConditionals))

# Total No conditionals
totalNoConditionals = 1  # can't be 0 to start
for item in testTweets[0]:
    if item in noDictionary:
        totalYesConditionals *= condNo(noDictionary[item])

totalNoConditionals *= math.log10(priorNo)
print("Score No : " + str(totalNoConditionals))

# Step 1 : Parse the tweet to have every word in a list.
# Step 2 : Get rid of words that are not in the finalDictionary
# Step 3 : For every words left, go find it's conditional probability for yesDictionary and NoDictionary

# P(word|class) = frequency of word in class x * 0.01 / number of words in class x + total number of words in dictionary

# def computeResult():
# scoreNo = log(priorNo) x log(P(word1 | No)) X log(P(word2 | No)) ... log(P(wordX | No))
# scoreYes = log(priorYes) x log(P(word1 | Yes)) X log((word2 | Yes)) ... log(P(wordX | Yes))
# return argmax (Score(yes), Score(no))
