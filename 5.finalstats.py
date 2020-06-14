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


df = pd.read_csv('sentiment.csv')
total=df['p_score'].count()
p=df[df['p_score']== 1 ].count()
nt=df[df['p_score']==0].count()
n=df[df['p_score']== -1 ].count()
print("positive %: ",(p['p_score']/total)*100)
print("neutral %: ",(nt['p_score']/total)*100)
print("negative %: ",(n['p_score']/total)*100)