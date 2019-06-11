# -*- coding: utf-8 -*-
import numpy as np
import nltk
import re
import pickle
from nltk.corpus import stopwords
from sklearn.datasets import load_files

#Importing Dataset
reviews = load_files('txt_sentoken/')
X,y=reviews.data,reviews.target


#Storing as pickle file:
with open('X.pickle','wb') as f :
pickle.dump(X,f)
with open('y.pickle','wb') as f :
pickle.dump(y,f)

#Unpickling the dataset
with open('X.pickle','rb') as f:
    p=pickle.load(f)
with open('y.pickle','rb') as f:
    y=pickle.load(f)

#Creating The corpus
corpus=[]
for i in range(0,len(X)):
    review = re.sub(r"\W",' ',str(X[i]))
    review =review.lower()
    review=re.sub(r"\s+[a-z0-9]\s+"," ",review)
    review=re.sub(r"^[a-z0-9]\s+"," ",review)
    review=re.sub(r"\s+"," ",review)
corpus.append(review)

#Creating the simple binary bag of model:
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(max_features =2000,min_df = 3,max_df =0.8,stop_words=stopwords.words('english'))
X=vectorizer.fit_transform(corpus).toarray()


#TF-IDF Model
from sklearn.feature_extraction.text import TfidfTransformer
transformer=TfidfTransformer()
X=transformer.fit_transform(X).toarray()

#creating training and test dataset
from sklearn.model_selection import train_test_split
text_train,text_test,sent_train,sent_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

#Logistic regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)

#Calculate the accuracy

sent_pred = classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test,sent_pred)

Accuracy =((cm[0][0]+cm[1][1])/4) 
print(Accuracy)

#pickling the classifier

with open('classifier.pickle','wb') as f:
pickle.dump(classifier,f)   
#pickiling the vectorizer
#Again Unpickling the dataset
with open('X.pickle','rb') as f:
    p=pickle.load(f)

with open('y.pickle','rb') as f:
    y=pickle.load(f)
#Again Creating The corpus
corpus=[]
for i in range(0,len(X)):
    review = re.sub(r"\W",' ',str(X[i]))
    review =review.lower()
    review=re.sub(r"\s+[a-z0-9]\s+"," ",review)
    review=re.sub(r"^[a-z0-9]\s+"," ",review)
    review=re.sub(r"\s+"," ",review)
corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(max_features =2000,min_df = 3,max_df =0.8,stop_words=stopwords.words('english'))
X=vectorizer.fit_transform(corpus).toarray()


with open ('tfidfmodel.pickle','wb') as f :
pickle.dump(vectorizer,f)

#unpickling the classifier & Vectorizer

with open ('classifier.pickle','rb') as f:
clf=pickle.load(f)
with open ('tfidfmodel.pickle','rb') as f:
tfidf =pickle.load(f)

#To check my classifier whether its work    
sample=["he is a bad boy"]
sample=tfidf.transform(sample).toarray()
sentiment=(clf.predict(sample))
if 0.5<=sentiment<=1 :
print("This is a positive sentence",sentiment)
else:
print("This is a negetivesentence",sentiment)
