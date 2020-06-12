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


df = pd.read_csv('final_sentiment.csv')
total=df['predicted_score'].count()
p=df[df['predicted_score']=='positive'].count()
nt=df[df['predicted_score']=='neutral'].count()
n=df[df['predicted_score']=='negative'].count()
print("positive %: ",(p['predicted_score']/total)*100)
print("neutral %: ",(nt['predicted_score']/total)*100)
print("negative %: ",(n['predicted_score']/total)*100)