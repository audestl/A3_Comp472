import csv
import math
from itertools import chain

from nltk import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


tsv_file = open("covid_training.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")
arr = []

for row in read_tsv:
    arr.append(row)

arr.remove(arr[0])
numrows = len(arr)
numcols = len(arr[0])

vocabulary = []
for i in range(numrows):
    arr[i][1] = arr[i][1].lower()
    vocabulary.append(arr[i][1])

vectorizer = CountVectorizer(stop_words='english')
vectorizer = CountVectorizer()
sparse_matrix = vectorizer.fit_transform(vocabulary)


doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix,
                  columns= vectorizer.get_feature_names())



# # Apply the model on the test set
tsv_file = open("covid_test_public.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")
testArr = []
for row in read_tsv:
    testArr.append(row)


# Writing the Trace file
f= open("trace_NB-BOW-OV.txt","w+")
for i in range(len(testArr)):
     f.write(testArr[i][0]+"  yes/no  classScore  correctClass  correct\n")

# Writing the Evaluation file
f= open("eval_NB-BOW-OV.txt","w+")
f.write("Accuracy\n"+"yes-Precision  no-Precision\n"+"yes-recall  no-recall\n"+"yes-F1  no-F1")

f.close()



