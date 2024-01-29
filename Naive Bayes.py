import numpy as np 
import pandas as pd 
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,f1_score

df = pd.read_csv("C:/school things/Courses/2023fall/Data and social media analysis/Finanal project/OLIDv1.0/olid-training-v1.0.tsv",sep='\t')
df.head()
df.info()

def data_processing(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\s+|www\S+http\S+", "",tweet,flags=re.MULTILINE)
    tweet = re.sub(r'\@w+|\#','',tweet)
    tweet = re.sub(r'[^\w\s]','',tweet)
    tweet = re.sub("user",'',tweet)
    tweet = re.sub("url",'',tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_tweets = [w for w in tweet_tokens if not w in stop_words]
    return " ".join(filtered_tweets)

df.tweet = df['tweet'].apply(data_processing)

vect = TfidfVectorizer(ngram_range=(1,3)).fit(df['tweet'])
X = df['tweet']
Y = df['subtask_a']

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
X_train = vect.transform(X_train).toarray()  # Convert to dense matrix
X_test = vect.transform(X_test).toarray() 
nb = GaussianNB()
nb.fit(X_train,y_train)
nb_predict = nb.predict(X_test)
nb_acc = accuracy_score(nb_predict,y_test)
print('Test accuarcy : {:.2f}%'.format(nb_acc*100))
print(round(f1_score(nb_predict, y_test, pos_label='OFF'), 2))
#Test accuarcy : 55.51%
#0.47