import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib import style
style.use('ggplot')
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

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
df = df.drop_duplicates("tweet")
df['subtask_a'].value_counts()
fig = plt.figure(figsize=(5,5))
sns.countplot(x = 'subtask_a',data=df)

fig = plt.figure(figsize=(7,7))
color = ("red","blue")
wp = {'linewidth':2, 'edgecolor': "black"}
tags = df['subtask_a'].value_counts()
explode = (0.1,0.1)
tags.plot(kind='pie',autopct = '%1.1f%%',shadow=True, colors = color, startangle = 90,
          wedgeprops = wp, explode= explode, label='')
plt.title('Distribution of sentiments')

non_offensive_tweets = df[df.subtask_a == 'NOT']
text = " ".join([word for word in non_offensive_tweets['tweet']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Most frequent words in non offensive tweets', fontsize = 19)
plt.show()

vect = TfidfVectorizer(ngram_range=(1,2)).fit(df['tweet'])
feature_names = vect.get_feature_names_out()
print('Number of features: {}\n'.format(len(feature_names)))
print('First 20 features : \n{}'.format(feature_names[:20]))

vect = TfidfVectorizer(ngram_range=(1,3)).fit(df['tweet'])
feature_names = vect.get_feature_names_out()
print('Number of features: {}\n'.format(len(feature_names)))
print('First 20 features : \n{}'.format(feature_names[:20]))

X = df['tweet']
Y = df['subtask_a']
X = vect.transform(X)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

print(f'size of x_train {x_train.shape}')
print(f'size of y_train {y_train.shape}')
print(f'size of x_test {x_test.shape}')
print(f'size of y_test {y_test.shape}')

svm = SVC(kernel= 'linear')
svm.fit(x_train,y_train)

svm_predict = svm.predict(x_test)
svm_acc = accuracy_score(svm_predict, y_test)
print(f'Test accuracy : {round(svm_acc,2)}')
print(f1_score(y_test, svm_predict, pos_label='OFF'))

#Test accuracy : 0.73
#F1 = 0.42
