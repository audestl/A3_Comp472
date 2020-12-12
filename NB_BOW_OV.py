import csv
from itertools import chain
import math
from nltk.tokenize import RegexpTokenizer

tsv_file = open("covid_training.tsv", encoding="UTF-8")
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
    for y in range(numcols):
        s = arr[i][1].split()
        vocabularyArr.append(s)

vocabularyArr = list(chain.from_iterable(vocabularyArr))

# To remove duplicates from the vocabulary
vocabularyArr = list(set(vocabularyArr))
seen = set()
finalVocabulary = []
for item in vocabularyArr:
    if item not in seen:
        seen.add(item)
        finalVocabulary.append(item)
print(finalVocabulary)
#
# # COMPUTE CONDITIONALS
# # Vocabulary = all words in the tweets of the training set
#
# # Step 1: Split the tweets based on yes or no
#
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
#
# # Step 2 : Count each instances of every word from the vocabulary in YesTweets
sizeYes = len(yesTweets)
#
for i in range(sizeYes):
    s = yesTweets[i].split()
    finalYesTweets.append(s)
finalYesTweets = list(chain.from_iterable(finalYesTweets))
#
# # Step 3 : Count each instances of every word from the vocabulary in NoTweets
sizeNo = len(noTweets)
for i in range(sizeNo):
    s = noTweets[i].split()
    finalNoTweets.append(s)
finalNoTweets = list(chain.from_iterable(finalNoTweets))
#
# # Count number of words in each dictionary
totalWordsNo = len(finalNoTweets)
totalWordsYes = len(finalYesTweets)
#
yesDictionary = dict()
noDictionary = dict()
# # Initialize "yes" tweets dictionary with training set
for i in range(len(finalVocabulary)):
    val = finalYesTweets.count(finalVocabulary[i])
    yesDictionary[finalVocabulary[i]] = val
#
# # Initialize "No" tweets dictionary with training set
for i in range(len(finalVocabulary)):
    val = finalNoTweets.count(finalVocabulary[i])
    noDictionary[finalVocabulary[i]] = val
#
#
#
# # COMPUTE PRIORS
# Calculate probabilities of each class (yes, no for factual tweet)
totalTweets = numTweetsNo + numTweetsYes
priorYes = numTweetsYes / totalTweets
priorNo = numTweetsNo / totalTweets
#
# # Apply the model on the test set

tsv_file = open("covid_test_public.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")
testArr = []
for row in read_tsv:
    testArr.append(row)

testTweets = []
for i in range(len(testArr)):
    s = testArr[i][1].split()
    testTweets.append(s)



def calculateCondYes(freq):
    prob = math.log10((freq + 0.01) / (sum(yesDictionary.values()) + (len(finalVocabulary) * 0.01)))
    return prob


def calculateCondNo(freq):
    prob = math.log10((freq + 0.01) / (sum(noDictionary.values()) + (len(finalVocabulary) * 0.01)))
    return prob


numCorrect = 0
numWrong = 0
fpYes = 0
fpNo = 0
fnYes = 0
fnNo = 0
conclusion = ""
modelPrediction = ""
f = open("trace_NB-BOW-OV.txt", "w+")
for i in range(len(testTweets)):
    # Total Yes conditionals
    scoreYes = 0  # can't be 0 to start
    scoreYes += math.log10(priorYes)
    for item in testTweets[i]:
        if item in yesDictionary:
            scoreYes += calculateCondYes(yesDictionary[item])

    # Total No conditionals
    scoreNo = 0  # can't be 0 to start
    scoreNo += math.log10(priorNo)
    for item in testTweets[i]:
        if item in noDictionary:
            scoreNo += calculateCondNo(noDictionary[item])

    if scoreNo > scoreYes:
        modelPrediction = "no"
        finalScore = scoreNo
    else:
        modelPrediction = "yes"
        finalScore = scoreYes

    if testArr[i][2] == "no" and modelPrediction == "yes":
        conclusion = "wrong"
        fpYes += 1
        fnNo += 1
        numWrong += 1
    elif testArr[i][2] == "yes" and modelPrediction == "no":
        conclusion = "wrong"
        fpNo += 1
        fnYes += 1
        numWrong += 1
    else:
        conclusion = "correct"
        numCorrect += 1

    f.write(str(testArr[i][0]) + "  " + str(modelPrediction) + "  " + str("{:e}".format(finalScore)) + "  " + str(
        testArr[i][2]) + "  " +
            str(conclusion) + "\n")


def calculateAccuracy():
    #     # % of instances of the test set the algorithm correctly
    return numCorrect / (numCorrect + numWrong)


print("Accuracy : " + str(calculateAccuracy()))


#
# TP = nb of instances that are in class C and that the model identified as C
# FP = nb of instances that the model labelled as class C
# FN = All instances that are in class C
def calculateRecallYes():
    return numCorrect / (numCorrect + fpNo)


def calculateRecallNo():
    return numCorrect / (numCorrect + fpNo)


def calculatePrecisionYes():
    return numCorrect / (numCorrect + fnYes)


def calculatePrecisionNo():
    return numCorrect / (numCorrect + fnNo)


def calculateF1Yes():
    return (2 * calculatePrecisionYes() * calculateRecallYes()) / (calculatePrecisionYes() + calculateRecallYes())


def calculateF1No():
    return (2 * calculatePrecisionNo() * calculateRecallNo()) / (calculatePrecisionNo() + calculateRecallNo())



f = open("eval_NB-BOW-OV.txt", "w+", encoding="UTF-8")
f.write(str(calculateAccuracy()) + "\n" + str(calculatePrecisionYes())
        + "  " + str(calculatePrecisionNo()) + "\n" + str(calculateRecallYes())
        + "  " + str(calculateRecallNo()) + "\n" + str(calculateF1Yes()) + "  " + str(calculateF1No()) + "\n")


f.close()
