import pandas as pd
import numpy as np
import csv
import re #regular expression
from textblob import TextBlob
import string
import preprocessor as p
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



#stops words
stop = stopwords.words('english')
#english words
english_word = set(nltk.corpus.words.words())
print('naam' in english_word)
#reading the dataset in csv file
df= pd.read_csv('hashtag_data.csv')
to_drop=['conversation_id','timezone','name','place','mentions','urls','photos',
         'cashtags','link','retweet','quote_url','video','near','geo','source',
         'user_rt_id','user_rt','retweet_id','reply_to','retweet_date','translate','trans_src',
         'trans_dest']

#drop columns
df.drop(to_drop, inplace=True, axis=1)
totallen=len(df)
print("Intial total length of the dataset",len(df))

#droping na records
print("Removing Na records")
df = df.dropna()
df = df.reset_index(drop=True)
print("NA records removed: ",totallen-len(df) )

#droping dulplicats records
print("Removing Duplicates records")
df = df.drop_duplicates()
#reset index after dropping
df = df.reset_index(drop=True)
print("duplicates records removed: ",totallen-len(df) )

print("converting string small case")
#convert text to small case
df['tweet'] = df['tweet'].str.lower()

print("Removing numbers...")
#removes numbers from text
df['tweet'] = df['tweet'].str.replace('\d+', '')
print("Numbers are removed")

print("Removing single character words")
#remove single character chracter
df['tweet'] = df['tweet'].replace(re.compile(r"(^| ).( |$)"), "")
print("Removing single character words")

print("Removing links")
#removes links and urls
df['tweet'] = df['tweet'].replace(re.compile(r'((www\.[\S]+)|(https?://[\S]+))'),"")
print("Links are removed")

print("Removing punchuation and special characters")
#removes puntuation
df['tweet'] = df['tweet'].str.replace('[^\w\s]',' ').str.replace('\s\s+', '')
print("Puntuations removed...")

print("Removing stop words...")
#remove stop words
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print("Stop words removed...")

print("Removing non english words")
#remove non english words
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word in (english_word)]))
print("Non english words removed")

df['tweet']=df['tweet'].str.strip()
df['tweet'] = df['tweet'].replace(re.compile(r"(^| ).( |$)"), " ")

print("Removing tweets having words less than 2 words")
#drops tweets less than 2 words
df.drop(df[df['tweet'].str.count(" ") < 3].index , inplace=True)
#reset index after dropping
df = df.reset_index(drop=True)
print("tweets having words less than 2 words are removed...")
print("word count less than 2 records removed: ",totallen-len(df) )


tokenized_tweet = df.tweet.apply(lambda x: x.split())
tokenized_tweet.head()
from nltk.stem.porter import * 
stemmer = PorterStemmer() 
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()


print("new data started writting in new csv file preprocessed_data.csv...")
#write clean data to new file
df.to_csv('preprocessed_data.csv', index=False, encoding="utf-8")
print("clean data is written on preprocessed_data.csv")

all_words = ' '.join([text for text in df['tweet']]) 

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 
plt.figure(figsize=(10, 7)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis('off')
plt.show()