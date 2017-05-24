# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
from time import time
import pandas,re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import utilitiesTrainOpt
import scipy.sparse as sps
import numpy as np
n_features = 1000
n_topics = 100
n_top_words = 5

ROWS = 100
COLS = 180
LATITUDE_UPPER_BOUND = 49.2827 + 2
LATITUDE_LOWER_BOUND = 49.2827 - 2
LONGITUDE_UPPER_BOUND = -123.1207 + 2
LONGITUDE_LOWER_BOUND = -123.1207 - 2
NW = 0.001

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        try:
          print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        except:
          print("Print Fail")
    print()

#I think here we can do data_samples = clean_text
pandaData = pickle.load(open('0.5pandaDataTraining.pkl','rb'))
pandaData = utilitiesTrainOpt.appendCoordinateGrid(pandaData,ROWS,COLS,NW,LATITUDE_UPPER_BOUND,LATITUDE_LOWER_BOUND, LONGITUDE_UPPER_BOUND,LONGITUDE_LOWER_BOUND)



raw_text = pandaData['text'].tolist()
# Make sure NaNs turn into strings
# (We probably don't want this in the long run)
raw_text = [str(x) for x in raw_text]
print("Number of Samples:", len(raw_text))
clean_text = [" ".join([   # joins a list of words back together with spaces in between them
                                re.sub(r'\W+', '', # force alphanumeric (after doing @ and # checks)
                                word.replace('"','').lower()) # force lower case, remove double quotes
                            for word in tweet.split() # go word by word and keep them if...
                                if len(word)>2 and # they are 3 characters or longer
                                #not word.startswith('@') and # they don't start with @, #, or http
                                #not word.startswith('#') and
                                not word.startswith('http')])
                            #.encode('ascii', errors='ignore') # force ascii encoding, ignore weird characters just in case
                        for tweet in raw_text]

xPos = list(pandaData["xgrid"])
yPos = list(pandaData["ygrid"])
#clean_text = [ (str(xPos[i])+str(yPos[i])+" ")+clean_text[i] for i in range(len(xPos))]
#print(clean_text[:20])
data_samples = clean_text



# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))



lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

#Next put testing tweets into each topic
#H is of size topic*words
#From feature_names[i] we know H
#(H*bagofwords matrix.T).T 

pickle.dump(lda,open("lda.pkl","wb"))
#############################################################
#Load after here

lda = pickle.load(open("lda.pkl","rb"))
#HTest = text_tf_idf
#H actually has to be something else
#print(HTest)



#Problem: Feature only takes 1000
#Fix: Truncate bag of words into only 1000 words
#For each word in features, take the column corresponding to those words, and truncate
vocab = pickle.load(open("0.5vocabDictBoth.pkl","rb"))
bagofwordsTest = pickle.load(open("0.5bagOfWordsTesting.pkl","rb"))
pandaDataTest = pickle.load(open("0.5pandaDataTestingCG.pkl","rb"))
indexToKeep = []
for word in tf_feature_names:
    indexToKeep.append(vocab[word])

bagofwordsTest = bagofwordsTest[:,indexToKeep] 
"""
Not sure if the order will be correct
"""
topics_guess = (sps.csr_matrix.dot(sps.csr_matrix(lda.components_) ,(sps.csr_matrix(bagofwordsTest).T))).T
TopicOfTweet = np.argmax(topics_guess.todense(),axis = 1)
pandaDataTest["topic"] = TopicOfTweet
#Test whether it's correct by finding airport topic and see