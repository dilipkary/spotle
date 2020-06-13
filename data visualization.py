## importing the dataset
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('vader_labelled_data.csv')
df.head()

df2 = df.groupby(df["p_score"],as_index=False).count()
plt.style.use('ggplot')
plt.figure(figsize=(6,6))
plt.bar(df2["p_score"],df2["hashtags"],color = ["red","blue","green"])
plt.title("Analysis of positive, negative and neutral hashtags")
plt.ylabel("Frequency")
plt.show()

##Visualization of data monthwise
df["date"] = pd.to_datetime(df["date"])
import datetime
import calendar
df["counter"] = 1;
df["month"] = pd.DatetimeIndex(df["date"]).month
df3 = df[["p_score","month","id"]].groupby(['p_score','month'],as_index=False).count()
positive = df3[df3["p_score"] == "positive"]
negative = df3[df3["p_score"] == "negative"]
neutral = df3[df3["p_score"] == "neutral"]
plt.subplots(1, figsize=(8, 6))
plt.plot(positive["month"],positive["id"],color="red")
plt.plot(negative["month"],negative["id"],color="blue")
plt.plot(neutral["month"],neutral["id"],color="black")
plt.legend(["positive", "negative","neutral"])
plt.title("Plot monthwise of positive, negative and neutral data")
plt.xlabel("Months")
plt.ylabel("Frequency")
plt.show()

## Weekwise analysis of the dataset
df["week"] = df['date'].dt.week
df4 = df[["p_score","week","id"]].groupby(["p_score","week"],as_index = False).count()
p_df = df4[df4["p_score"] == "positive"]
n_df = df4[df4["p_score"] == "negative"]
nn_df = df4[df4["p_score"] == "neutral"]
plt.subplots(1, figsize=(8, 6))
plt.plot(p_df["week"],p_df["id"],color="green")
plt.plot(n_df["week"],n_df["id"],color="red")
plt.plot(nn_df["week"],nn_df["id"],color="blue")
plt.legend(["positive", "negative","neutral"])
plt.title("Plot weekwise of positive, negative and neutral data")
plt.xlabel("Weeks")
plt.ylabel("Frequency")
plt.show()


