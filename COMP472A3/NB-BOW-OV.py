import numpy as np


def getWordCount():
    return len(vocabulary)

def printDictContents():
    for key, value in word_dict.items():
        print("\nKey: %s" % key)
        print("Value: %s" % value)


"""
Training:
1. Get the training set tweets. (set of tweets taht are already classified into the correct category)
2. Store the words into a dictionary with weighting scheme (using term frequency?)
3. Make a vector-space representation of every document
4. Use Multinomial NB model to classify a document into a pre-defined class 

Steps for Multinomial NB: Refer to spam filter slide #6 & Lecture Video 2.3 @ 15:50
* If a word appears in test set that was not in training set(in vocabluary) then we ignore that word.

1. Get how many tweets are true/fake P(fake) and P(true) i.e: P(true) = true_tweets/ total_tweets
2. Compute the conditionals/likelyhoods with smoothing      i.e: (frequency_word_appears_given_class + smoothing) / (total_words_given_class + smoothing)    

Testing:
1. check score of both classifiers i.e: Score(fale) = P(fake) * P(word1 | fake) * P(word2 | fake) * ... * P(wordn | fake)
2. assign class depending on whichever has a better score
"""


smoothing = 0.01
input = open("DataSet/covid_training.tsv","r", encoding="utf8")
rawText = input.read().lower()
lines = rawText.split("\n")

vocabulary = []
unsortedVocabulary = []
tweetList = []
totalFakeTweets = 0
totalFactualTweets = 0
word_dict = {}

#This takes all the lines from the raw text and saves each head word
for k in lines[1:-1]: # skipping headers
    wordCategory = 0 # 0 =  fake, 1 = Factual, -99 = corrupt
    listOfCols = k.split("\t")
    tweetList.append(listOfCols[1])
    #listOfCols[2] is the col with yes/no
    if listOfCols[2] == "no":
        totalFakeTweets += 1
        wordCategory = 0
    elif listOfCols[2] == "yes":
        totalFactualTweets += 1
        wordCategory = 1
    else:
        print("incomplete date")
        wordCategory = -99
    #add word to dict if its not existing
    wordList = listOfCols[1].split(" ")
    #Objects for words will be fake count, true count, totalCount;
    for word in wordList:
        if not word in word_dict:
            if wordCategory == 0:
                word_dict[word] = [1,0,1]
            elif wordCategory == 1:
                word_dict[word] = [0,1,1]
            else:
                print("incomplete data")
        else:
            word_instance = word_dict[word]
            if wordCategory == 0:
                word_instance[0] += 1
            else:
                word_instance[1] += 1
            word_instance[2] += 1


#save each word into a list
for tweet in tweetList:
    wordList = tweet.split(" ")
    #append word to the vocabulary if it is not a dupe
    for word in wordList:
        if word not in unsortedVocabulary:
            unsortedVocabulary.append(word)
            
printDictContents()

#sort the vocabulary
vocabulary = sorted(unsortedVocabulary)
total_tweets = totalFakeTweets + totalFactualTweets
totalWords = len(vocabulary)

print("Total factual tweets: " + str(totalFactualTweets))
print("Total fake tweets: " + str(totalFakeTweets))
print("Total tweets: " + str(total_tweets))
print("total words: "+ str(totalWords))

priorProbabilityFake = totalFakeTweets / total_tweets
priorProbabilityFactual = totalFactualTweets / total_tweets

print("======================================================================================")
print("prior prob of fake: " + str(priorProbabilityFake))
print("prior prob of factual: " + str(priorProbabilityFactual))


#Objects for words will be fake count, true count, totalCount;
#create dictionary



#create table
 

