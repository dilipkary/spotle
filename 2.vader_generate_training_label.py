import pandas as pd
import numpy as np
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
df = pd.read_csv('preprocessed_data.csv')
#df = pd.read_csv('newdata.csv')

""" score = analyser.polarity_scores(df['tweet'][0])
print(score['compound'],score)
df.drop(df[df['tweet'].str.count(" ") < 1 and ].index , inplace=True) """
#reset index after dropping
df = df.reset_index(drop=True)
def polararity_gen(row):
    text = row['tweet']
    score=analyser.polarity_scores(text)['compound']
    pl=''
    if score > 0.05:
        pl='positive'
    elif score <= 0.0 and text.count(" ") > 3:#-0.05:
        pl='negative'
    elif score >= 0 and score <= 0.05 :
        pl='neutral'
    
    return pl
df['p_score']=df.apply(polararity_gen,axis=1)
#df['p_score'] = df['tweet'].apply(lambda x: remove_punct(x))

total=df['p_score'].count()
p=df[df['p_score']=='positive'].count()
nt=df[df['p_score']=='neutral'].count()
n=df[df['p_score']=='negative'].count()
print("positive %: ",(p['p_score']/total)*100)
print("neutral %: ",(nt['p_score']/total)*100)
print("negative %: ",(n['p_score']/total)*100)
df.to_csv('vader_labelled_data.csv', index=False, encoding="utf-8")