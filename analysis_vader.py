import pandas as pd
import numpy as np
import csv
import re


from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

df = pd.read_csv('vader_labelled_data.csv')

total=df['p_score'].count()
p=df[df['p_score']=='positive'].count()
nt=df[df['p_score']=='neutral'].count()
n=df[df['p_score']=='negative'].count()
print("positive %: ",(p['p_score']/total)*100)
print("neutral %: ",(nt['p_score']/total)*100)
print("negative %: ",(n['p_score']/total)*100)
df['hashtags'] = df['hashtags'].str.replace('[^\w\s]',' ').str.replace('\s\s+', ' ')

s=""
for i in range(len(df)):
    s+=df.hashtags[i]+" "

#print(s)

# lower max_font_size, change the maximum number of word and lighten the background:
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(s)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()