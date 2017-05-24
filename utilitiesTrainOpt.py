import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, PCA, TruncatedSVD, LatentDirichletAllocation
import time
import re  # regex
import scipy.sparse as sps
import numpy as np
import pandas as pd
import math
from datetime import *
from collections import Counter
from numpy import unravel_index


#Panda -> precision == 10
def highestPrecision(pandaData):
	pandaDataHighestPrecision = pandaData[pandaData['gps_precision'] == 10]
	pickle.dump(pandaDataHighestPrecision, open('pandas_data_vanc_10.pkl','wb'))

"""
Anything from this point assumes that gps_precision == 10 is used
"""

#precision == 10 -> bagOfWordsMatrix
def getMatrixToNMF(pandaData):
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
                                not word.startswith('http')]
                            )#.encode('ascii', errors='ignore') # force ascii encoding, ignore weird characters just in case
                        for tweet in raw_text]


    stop_words = [] # stop words file includes English, Spanish, and Catalan
    with open('stop_words.txt','r') as f:
        stop_words = [word.replace("\n",'') for word in f.readlines()] # Have to remove \n's because I didn't copy the stop words cleanly

    print("Stop word examples:", stop_words[:10])

    print("\n----20 TWEETS----")
    # Lets make sure this looks right...
    for tweet in clean_text[:20]: # First 20 tweets!
        print(tweet) # the b before these means they are ascii encoded
    print("--------------")


    tf_idf = TfidfVectorizer(min_df=10,stop_words=stop_words, sublinear_tf= True)
    # min_df means ignore words that appear in less than that many tweets
    # we specify our stop words list here too

    #These parts are suspicious

    text_tf_idf = tf_idf.fit_transform(clean_text) # like we talked about,
    # fit_transform is short hand for doing a .fit() then a .transform()
    # because 2 lines of code is already too much I guess...

    print("Dumping feature names to disk...")
    pickle.dump(text_tf_idf, open('bagOfWordsMatrix.pkl', 'wb'))

"""
----------------------
LOCATION MATRIX PART
----------------------
"""
def GenerateGrid(rows, cols, neighbor_weight):
        #neighbor_weight = 0.5 ###first we create all of the location vectors
        nw = neighbor_weight
        vector_list = []
        for i in range(0,rows):
            for j in range(0, cols):
                A = np.zeros((rows,cols))
                for x in range(-1,2):
                    for y in range(-1,2):
                        if i+x<rows and i+x >-1 and j+y<cols and j+y >-1:
                            A[i+x][j+y]= nw
                A[i][j]=1
                B = sps.csr_matrix(A.reshape((1,rows*cols)))
                #print(B)
                vector_list.append(B)

        return vector_list


""""
OPT PLS
"""
def appendCoordinateGrid(pandaData,rows,cols,nw,laup,lalo, loup,lolo):
	pandaData = pandaData[pandaData["latitude"] < laup]
	pandaData = pandaData[pandaData["latitude"] > lalo]
	pandaData = pandaData[pandaData["longitude"] < loup]
	pandaData = pandaData[pandaData["longitude"] > lolo]
	
	maxlat = pandaData["latitude"].max()+10**(-12)
	minlat = pandaData["latitude"].min()-10**(-12)
	maxlong = pandaData["longitude"].max()+10**(-12)
	minlong = pandaData["longitude"].min()-10**(-12)

	

	print(minlat, maxlat)
	print(minlong, maxlong)

	
	scaleLat = rows/(maxlat-minlat)
	scaleLong = cols/(maxlong-minlong)
	#Remove for loop
	
	pandaData["xgrid"] = ((np.array(pandaData["longitude"]).astype(float)-minlong)*scaleLong).astype(int)
	pandaData["ygrid"] = ((np.array(pandaData["latitude"]).astype(float)-minlat)*scaleLat).astype(int)

	return pandaData


"""OPT PLS"""
def obtainLPart(pandaData, rows,cols,nw):
	vector_list = GenerateGrid(rows,cols,nw)
	coorlist = (np.array(pandaData["ygrid"])*cols) + np.array(pandaData["xgrid"])

	#remove for loop
	#coorlist = []
	# for i,row in pandaData.iterrows():
	# 	x = row["xgrid"]
	# 	y = row["ygrid"]
	# 	coorlist.append(y*cols + x)



	length = len(coorlist)
	#Gonna be tweet * location (rows*cols)
	Lpart = sps.vstack(vector_list[coorlist[i]] for i in range(0,length))
	return Lpart

"""Maybe gud enough?"""
def performLONMF(pandaData, rows, cols, nw, alpha, Lpart, nT, bagOfWords):

	Tpart = bagOfWords
	#print("shape of Tpart is", Tpart.shape)
	#print("shape of Lpart is", Lpart.shape)

	location_norm = sps.linalg.norm(Lpart, 'fro')
	text_norm = sps.linalg.norm(Tpart, 'fro')

	adjustedAlpha = alpha*(text_norm/location_norm) # Weight of location matrix, normalized so that text and location parts have the same frobinous norm
	Lpart = adjustedAlpha* Lpart
	NMFLOC = sps.hstack((Tpart,Lpart))
	NMFLOC = NMFLOC.tocsr()
	print(NMFLOC.shape)

	######## PYTHON NMF #############
	topic_model = NMF(n_components=nT, verbose=1, tol=0.005)  # Sure lets compress to 100 topics why not...

	text_topic_model_W = topic_model.fit_transform(NMFLOC) # NMF's .transform() returns W by
	# default, but we can get H as follows:
	text_topic_model_H = topic_model.components_

	text_topic_model_WH = (text_topic_model_W,text_topic_model_H)
	
	return text_topic_model_WH


def putTimeInBox(hourList):
	ans = list(range(24))
	for i in range(len(ans)):
		ans[i] = []
	for i in range(len(hourList)):
		ans[hourList[i]].append(i)
	for i in range(len(ans)):
		ans[i] = len(ans[i])
	return ans
def putDateInBox(dateArray):
	dateList = list(set(dateArray))
	dateList.sort()
	keys = Counter(dateArray).keys()
	values = Counter(dateArray).values()
	countDict = {}
	keys = list(keys)
	values = list(values)
	maxValue = max(values)
	for i in range(len(keys)):
		countDict[keys[i]] = values[i]
	dateCount = []
	for i in range(len(keys)):
		dateCount.append(countDict[dateList[i]])
		if (countDict[dateList[i]] == maxValue):
			bestDate = dateList[i]
	return dateCount,bestDate


"""OPT PLS"""
def analysisNMF(pandaData,number_of_topics,W,H,cols,rows):
	Topics = W.argmax(axis = 1)
	pandaData["topics"] = Topics.tolist()
	#Add their MSD here
	NT = number_of_topics
	MSD_List = []#stores the Mean Square Distance between all the tweets in  agiven topic
	Topics_Size = []
	inFractionList = []
	dateBoxTest = []
	#Spatial = Spatial[Spatial["gps_precision"] == 10.0]#taking only location accurete tweets
	for T in range(0,NT):#Mean Square Distance Calculation: we want to find the sum of the square of the
	    # x =	pandaData[pandaData["topics"] == T]# euclidean pairwise distances between all the tweets in each topic divided by the size of the topic (K)
	    # a = x["latitude"]
	    # b = x["longitude"]
	    # K =len(a)
	    # Topics_Size.append(K)
	    # X = np.array(a)#we calculate x axis pairwise square distances
	    # Y = np.array(b)#and then seperatley y axis distances
	    # #Drop the square, see what happens
	    # MSD = (2/((K+1)))*((K*np.dot(X.T, X) - (X.sum())**2)+(K*np.dot(Y.T, Y) - (Y.sum())**2))#I am almost 100% this is correct but tell me if you guys see anything alarming
	    # MSD_List.append(MSD)#in the above, we used K+1 instead of K (Where K is the size of the topic) in the denomenator to make  sure we ar enot dividing by 0 for the empty topics
	    
	
		# x = a
		# timeArrayTopic = np.array(x["time"])
		# dateList = []
		# hourList = []
		# for i in range(len(timeArrayTopic)):
		# 	A = datetime.strptime(timeArrayTopic[i],'%Y-%m-%d %H:%M:%S')
		# 	dateList.append(A.date())
		# 	hourList.append(A.hour)
		# hourBox = putTimeInBox(hourList)
		# dateBox,bestDate = putDateInBox(dateList)
		# print("Length of datebox is", len(dateBox))
		# maxDateBox = max(dateBox)
		# dateBox.sort()
		# hourBox.sort()



		# if (len(dateBox) > 1):
		# 	if (dateBox[-1] >= 3.5*dateBox[-2]):
		# 		print("Nice!!")
		# 		dateBoxTest.append([True,bestDate])
		# 	else:
		# 		dateBoxTest.append([False])
		# else:
		# 	dateBoxTest.append([False])
		a = pandaData[pandaData["topics"] == T]
		K = len(a)
		Topics_Size.append(K)
		x = np.array(a["xgrid"])
		y = np.array(a["ygrid"])
	    
		countGood = 0
		K = int(K/2)
		if (K >= 1000):
			K = int(K/5)

		for i in range(int(K)):
			for j in range(int(K)):
				if ((x[i] - x[j])^2 + (y[i] - y[j])^2 <= 2):
					countGood += 1
		inFractionList.append(countGood/((K)+0.001)**2)
	    
	ArrayList = []

	for T in range(0,NT):#F density calculation
	    A = np.zeros((cols,rows))
	    G = pandaData[pandaData["topics"] == T]
	    N = len(G.index)
	    for row in G.itertuples():
	        x = row[10]
	        y= row[11]
	        A[x,y] = A[x,y]+ 1
	    ArrayList.append(A/N)
	TopicPeak = []
	for T in range(0,NT):
	    B = ArrayList[T]
	    peak = unravel_index(B.argmax(), B.shape)
	    TopicPeak.append([peak[0],peak[1]])

	print(TopicPeak[0])
	df = pd.DataFrame(columns = ( "Length", "inFraction"))#save the topics size and MSD into a data frame
	#print("Length of infractionlist is ", len(inFractionList))
	#print("Length of topicpeak is ", len(TopicPeak))
	df["inFraction"] = inFractionList
	df["peak"]= TopicPeak
	#df["MSD"] = MSD_List
	df["Length"] = Topics_Size
	#df["DateBox"] = dateBoxTest
	
	#HERE!!!
	#See if those topics are good or not
	#NOW HERE
	#ADD MSD TO EACH column
	#pickle.dump(pandaData, open('pandas_data_vanc_10_CG_t.pkl','wb'))
	return df,pandaData
	


