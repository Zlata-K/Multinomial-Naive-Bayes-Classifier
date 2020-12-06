import numpy as np

#just used for debugging, can remove when done with the assignment
def getWordCount():
    return len(vocabulary)

#print values, used for debugging
def printDictContents():
    for key, value in word_dict.items():
        print("\nKey: %s" % key)
        print("Value: %s" % value)

#function to add smoothing to all the words.
def smoothDictContents():
    for key, value in word_dict.items():

        fakeCounter = value[0]+smoothing
        factualCounter = value[1]+smoothing
        totalCounter = value[2]+smoothing
        word_dict[key] = [fakeCounter, factualCounter, totalCounter]

#tweetList and which class to check for are the params
#Objects for words will be fake_count, true_count, totalCount; therefore yes ends up being index 1. No is index 0
def getScore(tweetList, classification):
    if classification == "yes":
        index_to_check = 1
        totalityOfTweets = smoothedTotalFactualTweets
        score = np.log10(totalityOfTweets / smoothedTotalTweets) # getting prior
    else:
        index_to_check = 0
        totalityOfTweets = smoothedTotalFakeTweets
        score = np.log10(totalityOfTweets / smoothedTotalTweets) #getting prior 

    for word in tweetList:
        if word in word_dict:
            #TODO: is it += or *= here? This might be why its 99.5% accurate
            score += np.log10(word_dict[word][index_to_check] / totalityOfTweets) #calculate full score 
    return score

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

#given smoothing value
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

print("======================================================================================")
print("Before Smoothing")
print("Total factual tweets: " + str(totalFactualTweets))
print("Total fake tweets: " + str(totalFakeTweets))
print("Total tweets: " + str(total_tweets))
print("total words: "+ str(totalWords))

priorProbabilityFake = totalFakeTweets / total_tweets
priorProbabilityFactual = totalFactualTweets / total_tweets

print("======================================================================================")
print("After Smoothing")
smoothedTotalFactualTweets = totalFactualTweets + (smoothing * totalWords)
smoothedTotalFakeTweets = totalFakeTweets + (smoothing * totalWords)
smoothedTotalTweets = smoothedTotalFakeTweets + smoothedTotalFactualTweets
print("Total smoothed factual tweets: " + str(smoothedTotalFactualTweets))
print("Total smoothed fake tweets: " + str(smoothedTotalFakeTweets))

print("======================================================================================")
print("prior prob of fake: " + str(priorProbabilityFake))
print("prior prob of factual: " + str(priorProbabilityFactual))

smoothDictContents() #smoothing in order to avoid -infinity
printDictContents() #check the contents of dict to verify if it worked.

#get testing tweets:
test_input = open("DataSet/covid_test_public.tsv","r", encoding="utf8")
test_rawText = test_input.read().lower()
test_lines = test_rawText.split("\n")

#This takes all the lines from the raw text and saves each head word
counter = 0 ;
f = open("trace_NB-BOW-OV.txt", "w")
f.close()

totalCorrectPredictions = 0 
totalWrongPredictions = 0

for k in lines[1:-1]: # skipping headers of the tsv file
    listOfCols = k.split("\t")
    tweetID = listOfCols[0]
    testing_wordList = listOfCols[1].split(" ") #seperate the words in a tweet into a list
    correctAnswer = listOfCols[2] #the correct classification

    #Get both and check which one is the biggest
    factualScore = getScore(testing_wordList, "yes")
    fakeScore = getScore(testing_wordList, "no")

    print("==================================================================")
    print(str(counter))
    print(str(factualScore))
    print(str(fakeScore))
    counter += 1

    #Take the largest score
    if factualScore > fakeScore:
        finalScore = np.format_float_scientific(factualScore, precision=3)
        prediction = "yes"
    else:
        finalScore = np.format_float_scientific(fakeScore,  precision=3)
        prediction = "no"
    print("Final score")
    print(finalScore)

    if prediction == correctAnswer:
        isRight = "correct" 
        totalCorrectPredictions += 1
    else:
        isRight = "wrong"
        totalWrongPredictions += 1

    f = open("trace_NB-BOW-OV.txt", "a")
    f.write(tweetID+ "  " + prediction + "  " + finalScore + "  " + correctAnswer + "  "+ isRight + "\n")
    f.close()

    ## TODO: why isthis 99.5% success rate?


#evaluation info
totalPredictions = totalCorrectPredictions + totalWrongPredictions
accuracy = totalCorrectPredictions / totalPredictions

#evaluation file clean new file
f = open("eval_NB-BOW-OV.txt", "w")
f.close()

#evaluation file append to newly created file
print("Accuracy is: "+  str(accuracy))
f = open("eval_NB-BOW-OV.txt", "a")
f.close()