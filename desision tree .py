import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from nltk.tokenize import word_tokenize
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import pr
from sklearn.metrics import accuracy_score,f1_score

stemmer = nltk.SnowballStemmer('english')
stopword = set(stopwords.words('english'))

df = pd.read_csv("C:/school things/Courses/2023fall/Data and social media analysis/Finanal project/OLIDv1.0/olid-training-v1.0.tsv", sep='\t')
df.head()

def data_processing(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\s+|www\S+http\S+", "", tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@w+|\#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub("user", '', tweet)
    tweet = re.sub("url", '', tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_tweets = [stemmer.stem(w) for w in tweet_tokens if w.lower() not in stopword and w.isalpha()]
    return " ".join(filtered_tweets)

df['tweet'] = df['tweet'].apply(data_processing)

df.head()
vect = TfidfVectorizer(ngram_range=(1,3)).fit(df['tweet'])
X = df['tweet']
Y = df['subtask_a']

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
X_train = vect.transform(X_train)
X_test = vect.transform(X_test)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf_predict = clf.predict(X_test)
clf_acc = accuracy_score(clf_predict,y_test)
print('Test accuarcy : {:.2f}%'.format(clf_acc*100))
print(round(f1_score(clf_predict, y_test, pos_label='OFF'), 2))
#Test accuarcy : 72.66%
#0.57

#Chinese Logistic Baseline
df_cn = pd.read_csv("C:/school things/Courses/2023fall/Data and social media analysis/Finanal project/COLD/train.csv",sep=',')

vect = TfidfVectorizer(ngram_range=(1,3)).fit(df_cn['TEXT'])

X_1 = df_cn['TEXT']
Y_1 = df_cn['label']
X_1 = vect.transform(X_1)

x_train_cn,x_test_cn,y_train_cn,y_test_cn = train_test_split(X_1,Y_1, test_size= 0.2,random_state=42)

clf = DecisionTreeClassifier()

clf.fit(x_train_cn,y_train_cn)
clf_predict_cn = clf.predict(x_test_cn)
clf_acc_cn = accuracy_score(clf_predict_cn, y_test_cn)
print('Test accuarcy : {:.2f}%'.format(clf_acc_cn*100))
print(round(f1_score(y_test_cn, clf_predict_cn,pos_label=1),2))
#Test accuarcy : 54.99%
#0.23