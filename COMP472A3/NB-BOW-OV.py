import numpy as np

input = open("DataSet/covid_training.tsv","r", encoding="utf8")
rawText = input.read().lower()
lines = rawText.split("\n")

vocabulary = []
tweetList = []


for k in lines[1:-1]: # skipping headers
    listOfCols = k.split("\t")
    tweetList.append(listOfCols[1])

for tweet in tweetList:
    wordList = tweet.split(" ")


    for word in wordList:
        if word not in vocabulary:
            vocabulary.append(word)

print(vocabulary)