import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

df = pd.read_csv('vader_labelled_data.csv')

def polararity_gen(row):
    text = row['p_score']
    
    pl=0
    if text=='positive':
        pl=1
    elif text =='negative':
        pl=-1
    elif text =='neutral':
        pl=0
    
    return pl
df['p_score']=df.apply(polararity_gen,axis=1)
df.to_csv('sentiment.csv', index=False, encoding="utf-8")
