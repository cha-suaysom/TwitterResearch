# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
from time import time
import pandas,re
import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import utilitiesTrainOpt
import scipy.sparse as sps
import numpy as np
n_features = 1000
n_topics = 100
n_top_words = 10

ROWS = 100
COLS = 180
LATITUDE_UPPER_BOUND = 49.2827 + 2
LATITUDE_LOWER_BOUND = 49.2827 - 2
LONGITUDE_UPPER_BOUND = -123.1207 + 2
LONGITUDE_LOWER_BOUND = -123.1207 - 2
NW = 0.001
gridlength = 260

def print_top_words(model, feature_names, n_top_words):
    topicLocationX = []
    topicLocationY = []
    locationValue = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        try:
          print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
          x=1
        except:
          print("Print Fail")

        for i in topic.argsort()[::-1]:
            try:
                if (int(feature_names[i]) >= 11):
                    #print("The location is", feature_names[i])
                    xLoc = int(feature_names[i][:2])
                    yLoc = int(feature_names[i][2:]) 
                    topicLocationX.append(xLoc)
                    topicLocationY.append(yLoc)
                    break

            except:
                continue
        locationTopic = []
        for i in range(len(topic)):
            try:
                if (int(feature_names[i]) >= 11):
                    locationTopic.append(topic[i])
            except:
                continue
        locationValue.append(locationTopic)

    return topicLocationX,topicLocationY,locationValue

# #I think here we can do data_samples = clean_text
# pandaData = pickle.load(open('0.5pandaDataTraining.pkl','rb'))
# pandaData = utilitiesTrainOpt.appendCoordinateGrid(pandaData,ROWS,COLS,NW,LATITUDE_UPPER_BOUND,LATITUDE_LOWER_BOUND, LONGITUDE_UPPER_BOUND,LONGITUDE_LOWER_BOUND)



# raw_text = pandaData['text'].tolist()
# # Make sure NaNs turn into strings
# # (We probably don't want this in the long run)
# raw_text = [str(x) for x in raw_text]
# print("Number of Samples:", len(raw_text))
# clean_text = [" ".join([   # joins a list of words back together with spaces in between them
#                                 re.sub(r'\W+', '', # force alphanumeric (after doing @ and # checks)
#                                 word.replace('"','').lower()) # force lower case, remove double quotes
#                             for word in tweet.split() # go word by word and keep them if...
#                                 if len(word)>2 and # they are 3 characters or longer
#                                 #not word.startswith('@') and # they don't start with @, #, or http
#                                 #not word.startswith('#') and
#                                 not word.startswith('http')])
#                             #.encode('ascii', errors='ignore') # force ascii encoding, ignore weird characters just in case
#                         for tweet in raw_text]

# xPos = list(pandaData["xgrid"])
# yPos = list(pandaData["ygrid"])
# clean_text = [ (str(xPos[i])+str(yPos[i])+" ")+clean_text[i] for i in range(len(xPos))]
# #print(clean_text[:20])
# data_samples = clean_text



# # Use tf (raw term count) features for LDA.
# print("Extracting tf features for LDA...")
# tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
#                                 max_features=n_features,
#                                 stop_words='english')
# t0 = time()
# tf = tf_vectorizer.fit_transform(data_samples)
# print("done in %0.3fs." % (time() - t0))



# lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
#                                 learning_method='online',
#                                 learning_offset=50.,
#                                 random_state=0)
# lda.fit(tf)

# print("\nTopics in LDA model:")
# tf_feature_names = tf_vectorizer.get_feature_names()




# #Next put testing tweets into each topic
# #H is of size topic*words
# #From feature_names[i] we know H
# #(H*bagofwords matrix.T).T 

# pickle.dump(lda,open("lda.pkl","wb"))
# pickle.dump(tf_feature_names,open("tf_feature_names.pkl","wb"))
#############################################################
#Load after here

lda = pickle.load(open("lda.pkl","rb"))
tf_feature_names = pickle.load(open("tf_feature_names.pkl","rb"))
#HTest = text_tf_idf
#H actually has to be something else
#print(HTest)
topicLocationX,topicLocationY,locationTopic = print_top_words(lda, tf_feature_names, n_top_words)


#Problem: Feature only takes 1000
#Fix: Truncate bag of words into only 1000 words
#For each word in features, take the column corresponding to those words, and truncate
vocab = pickle.load(open("0.5vocabDictBoth.pkl","rb"))
bagofwordsTest = pickle.load(open("0.5bagOfWordsTesting.pkl","rb"))
pandaDataTest = pickle.load(open("0.5pandaDataTestingCG.pkl","rb"))
indexToKeep = []
take = 0

for i in range(len(tf_feature_names)):
    if (take == 1):
        indexToKeep.append(vocab[tf_feature_names[i]])
    if (tf_feature_names[i] == "9927"):
        take = 1
        index = i

bagofwordsTest = bagofwordsTest[:,indexToKeep] 
"""
Not sure if the order will be correct
"""
topics_guess = (sps.csr_matrix.dot(sps.csr_matrix(lda.components_[:,index+1:]) ,(sps.csr_matrix(bagofwordsTest).T))).T
TopicOfTweet = np.argmax(topics_guess.todense(),axis = 1)
projectionValue = np.max(sklearn.preprocessing.normalize(topics_guess.todense()),axis = 1)
#projectionValue = sklearn.preprocessing.normalize(projectionValue) 
pandaDataTest["projection"] = projectionValue
pandaDataTest["topic"] = TopicOfTweet
pandaDataTest["predictedCoordinateX"] = np.array(topicLocationX)[np.array(TopicOfTweet)]
pandaDataTest["predictedCoordinateY"] = np.array(topicLocationY)[np.array(TopicOfTweet)]



MSD = np.var(locationTopic,axis=1)
"""
Also need to filter based on the prediction
"""
numTopicPredict = 100
topicToPredict = np.argsort(MSD)[:(-1)*numTopicPredict-1:-1]


pandaDataTestAll = pandaDataTest

def doPrediction(pandaDataTest):
    xDistance = (np.array(pandaDataTest["xgrid"])-np.array(pandaDataTest["predictedCoordinateX"]))**2
    yDistance = (np.array(pandaDataTest["ygrid"])-np.array(pandaDataTest["predictedCoordinateY"]))**2
    len(xDistance)

    dis = (xDistance + yDistance)*gridlength 
    numberOfPredict = len(dis)
    print("We are predicting", numberOfPredict)
    print("Exact", len(dis[dis == 0])*100/numberOfPredict)
    print("HalfKM", len(dis[dis < 500])*100/numberOfPredict)
    print("OneKM", len(dis[dis < 1000])*100/numberOfPredict)
    print("TwoKM", len(dis[dis < 2000])*100/numberOfPredict)

for i in topicToPredict:
    print("predict topic ",i)
    pandaDataTest = pandaDataTestAll[pandaDataTestAll["topic"] == i]
    pandaDataTest = pandaDataTest[pandaDataTest["projection"] > 0.95]
    doPrediction(pandaDataTest)



#Test whether it's correct by finding airport topic and see

