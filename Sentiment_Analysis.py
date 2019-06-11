# -*- coding: utf-8 -*-
import tweepy
import re
import pickle
import pandas as pd
import numpy as np
import tweepy           # To consume Twitter's API
import pandas as pd     # To handle data
import numpy as np      # For number computing

# For plotting and visualization:
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

from tweepy import OAuthHandler

# Twitter App access keys for @user

# Consume:
CONSUMER_KEY    = 'sx6llfl3gRsOBNw3y7VVKSQT7'
CONSUMER_SECRET = 'DSZiX1mHEsKTcT7JMnahQY2dPC4lQNV21hkorPszWkz8D2Q5oH'

# Access:
ACCESS_TOKEN  = '998804133288144901-pPeZ4V3B9jmEDhhVbvML9tYAMgXy55A'
ACCESS_SECRET = 'AdqyikZTdUaStqYAF4hKzCpRwSiLHWveOjlygarnrFIBY'

# Authentication and access using keys:
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

args=['Facebook']
api = tweepy.API(auth,timeout=10)

list_tweets=[]
query=args[0]
if len(args)==1:
    for status in tweepy.Cursor(api.search,q=query+" -filter:retweets",lang='en',result_type='recent').items(500):
list_tweets.append(status.text)

# We create a pandas dataframe as follows:
data= pd.DataFrame(data=[tweet for tweet in list_tweets], columns=['Tweets'])

print('We display the first 10 elements of the dataframe:')
display(data.head(10))

print('Internal methods of a single tweet object:')
print(dir(list_tweets[0]))


# We add relevant data:
data['len']  =np.array([len(tweet) for tweet in list_tweets])

# We extract the mean of lenghts:
mean = np.mean(data['len'])
print("The average length of Tweets",mean)
#Loading TF-IDF model & Classifier

with open ('tfidfmodel.pickle','rb') as f:
    vectorizer =pickle.load(f)
with open ('classifier.pickle','rb') as f:
clf=pickle.load(f)

#Preprocessing The tweets

total_pos = 0
total_neg = 0

for tweet in list_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s"," ",tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]\s+"," ",tweet)
    tweet = re.sub(r"\s+http://t.co/[a-zA-Z0-9]*$"," ",tweet)
    tweet=tweet.lower()
    tweet=re.sub(r"rt"," ",tweet)
    tweet=re.sub(r'@[^\s]+',' ',tweet)
    tweet=re.sub(r"that's","thatis",tweet)
    tweet=re.sub(r"there's","thereis",tweet)
    tweet=re.sub(r"what's","whatis",tweet)
    tweet=re.sub(r"it's","itis",tweet)
    tweet=re.sub(r"who's","whois",tweet)
    tweet=re.sub(r"i'm","iam",tweet)
    tweet=re.sub(r"she's","sheis",tweet)
    tweet=re.sub(r"he's","heis",tweet)
    tweet=re.sub(r"they're","theyare",tweet)
    tweet=re.sub(r"who're","whoare",tweet)
    tweet=re.sub(r"ain't","amnot",tweet)
    tweet=re.sub(r"don't","donot",tweet)
    tweet=re.sub(r"doesn't","doesnot",tweet)
    tweet=re.sub(r"didn't","didnot",tweet)
    tweet=re.sub(r"wouldn't","wouldnot",tweet)
    tweet=re.sub(r"shouldn't","shouldnot",tweet)
    tweet=re.sub(r"can't","cannot",tweet)
    tweet=re.sub(r"isn't","isnot",tweet)
    tweet=re.sub(r"it's","it is not",tweet)
    tweet=re.sub(r"isn't","isnot",tweet)
    tweet=re.sub(r"wasn't","wasnot",tweet)
    tweet=re.sub(r"weren't","werenot",tweet)
    tweet=re.sub(r"couldn't","couldnot",tweet)
    tweet=re.sub(r"won't","willnot",tweet)
    tweet=re.sub(r"\W"," ",tweet)
    tweet=re.sub(r"\d"," ",tweet)
    tweet=re.sub(r"\s+[a-zA-Z]\s+"," ",tweet)
    tweet=re.sub(r"\s+[a-zA-Z]$"," ",tweet)
    tweet=re.sub(r"^[a-z]\s+"," ",tweet)
    tweet=re.sub(r"https"," ",tweet)
    tweet=re.sub(r"http\s+","",tweet)
    tweet=re.sub(r"yifmqy"," ",tweet)
    tweet=re.sub(r"\s+"," ",tweet)
    tweet=tweet.strip('\'"')
#    import fileinput
#    for line in fileinput.input(inplace=1):
#        line = line.rstrip()
#        tweet = fileinput.lineno()
#        print ('%-40s # %2i' % (line, tweet))
#    print( tweet)
#predicting sentiment
    sentiment=clf.predict(vectorizer.transform([tweet]).toarray())
    print("-"+tweet,":",sentiment)


    if sentiment[0]==1:
total_pos +=1
    else:
total_neg +=1

a=total_pos
b=total_neg
print("Total Positive tweet from extracting tweet=",a)
print("Total Negative tweet from extracting tweet=",b)


#Accuracy =((sentiment[0][0]+sentiment[1][1])/4) 
#print(Accuracy)

#describe_DataFrame=tweet.describe()
#print("Different Properties of  for Iphone",describe_DataFrame)


#plotting the result of sentiments

import matplotlib.pyplot as plt
import numpy as np 
objects=['Positive','Negative']
y_pos=np.arange(len(objects))
plt.bar(y_pos,[total_pos,total_neg],alpha=0.5)
plt.xticks(y_pos,objects)
plt.ylabel('Number')
plt.title('Number of positive &negetive tweets')
plt.show()

#Pie-Chart

labels = 'Positive', 'Negative'
sizes = [total_pos,total_neg]
colors = ['blue', 'red']


## use matplotlib to plot the chart
plt.pie(sizes, labels = labels, colors = colors, shadow = True, startangle = 90)
plt.title("Sentiment of 100 Tweets about Iphone")
plt.show()
