import csv
from itertools import chain
import math
from collections import defaultdict

file1 = open("covid_training.tsv", 'r', encoding="utf-8")
Tweets = file1.readlines()

vocabDictionary = defaultdict(int)
filteredVocabDictionary = defaultdict(int)

yesTweets = []
noTweets = []
yesDictionary = defaultdict(int)
noDictionary = defaultdict(int)

for tweet in Tweets:
    # treatment of the tweet to lowercase and remove id and useless classes
    tweet = tweet.lower()
    tempArray = tweet.split()
    tempArray = tempArray[1:-6]
    # Check classification of the tweet and add it to the correct list
    if tempArray[len(tempArray) - 1] == "yes":
        tempArray.pop(len(tempArray) - 1)
        yesTweets.append(" ".join(tempArray))
    else:
        tempArray.pop(len(tempArray) - 1)
        noTweets.append(" ".join(tempArray))
    for elem in tempArray:
        vocabDictionary[elem] += 1

#Create dictionaries of word occurences in the tweet lists
for item in yesTweets:
    for word in item.split():
        yesDictionary[word] += 1

for item in noTweets:
    for word in item.split():
        noDictionary[word] += 1

#Filter the dictionaries
vocabDictionary = {key: val for key, val in vocabDictionary.items() if val != 1}
yesDictionary = {key: val for key, val in yesDictionary.items() if val != 1}
noDictionary = {key: val for key, val in noDictionary.items() if val != 1}

# COMPUTE PRIORS
# Calculate probabilities of each class (yes, no for factual tweet)
totalTweets = len(yesTweets) + len(noTweets)
priorYes = len(yesTweets) / totalTweets
priorNo = len(noTweets) / totalTweets

# Apply the model on the test set
tsv_file = open("covid_test_public.tsv", 'r', encoding="utf-8")
Tweets2 = tsv_file.readlines()
testTweets = []

#Treatment of the Test tweets
for tweet in Tweets2:
    tweet = tweet.lower()
    tempArray = tweet.split()
    tempArray = tempArray[:-6]
    testTweets.append(" ".join(tempArray))

def calculateCondYes(freq):
    prob = math.log10((freq + 0.01) / (sum(yesDictionary.values()) + (len(vocabDictionary) * 0.01)))
    return prob


def calculateCondNo(freq):
    prob = math.log10((freq + 0.01) / (sum(noDictionary.values()) + (len(vocabDictionary) * 0.01)))
    return prob

numCorrect = 0
numWrong = 0
fpYes = 0
fpNo = 0
fnYes = 0
fnNo = 0
conclusion = ""
modelPrediction = ""
f = open("trace_NB-BOW-FV.txt", "w+")
for item in testTweets:
    # Total Conditionals
    print(item)
    scoreYes = math.log10(priorYes)
    scoreNo = math.log10(priorNo)
    for word in item.split():
        print(word)
        if word in vocabDictionary:
            if word in yesDictionary:
                scoreYes += calculateCondYes(yesDictionary[word])
            else:
                scoreYes += calculateCondYes(0)
            if word in noDictionary:
                scoreNo += calculateCondNo(noDictionary[word])
            else:
                scoreNo += calculateCondNo(0)


    if scoreNo > scoreYes:
        modelPrediction = "no"
        finalScore = scoreNo
    else:
        modelPrediction = "yes"
        finalScore = scoreYes

    id, *tweet, classification = item.split()

    if classification == "no" and modelPrediction == "yes":
        conclusion = "wrong"
        fpYes += 1
        fnNo += 1
        numWrong += 1
    elif classification == "yes" and modelPrediction == "no":
        conclusion = "wrong"
        fpNo += 1
        fnYes += 1
        numWrong += 1
    else:
        conclusion = "correct"
        numCorrect += 1

    # output to trace file
    f.write(id + "  " + str(modelPrediction) + "  " + str("{:e}".format(finalScore)) + "  "
           + classification + "  " +str(conclusion) + "\n")


def calculateAccuracy():
    # % of instances of the test set the algorithm correctly
    return numCorrect / (numCorrect + numWrong)


print("Accuracy : " + str(calculateAccuracy()))

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

#output to eval file
f = open("eval_NB-BOW-FV.txt", "w+", encoding="UTF-8")
f.write(str(calculateAccuracy()) + "\n" + str(calculatePrecisionYes())
        + "  " + str(calculatePrecisionNo()) + "\n" + str(calculateRecallYes())
        + "  " + str(calculateRecallNo()) + "\n" + str(calculateF1Yes()) + "  " + str(calculateF1No()) + "\n")

f.close()
