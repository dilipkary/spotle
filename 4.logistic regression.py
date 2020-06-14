import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

df = pd.read_csv('sentiment.csv')
print("Columns in the original dataset:\n")
print(df.columns)


from sklearn.model_selection import train_test_split

X = df['tweet']
y = df['p_score']

one_hot_encoded_label = pd.get_dummies(y)
print(one_hot_encoded_label.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

#second method
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df=5).fit(X_train)
print(len(vect.get_feature_names()))
X_train_vectorized = vect.transform(X_train)
feature_names = np.array(vect.get_feature_names())
sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()

print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


#second method
X_train_vectorized = vect.transform(X_train)
from sklearn.metrics import accuracy_score

c_val =[0.2,0.75, 1, 2, 3, 4, 5, 10,14,19,20]
for c in c_val:
    model = LogisticRegression(C=c,solver='saga')
    model.fit(X_train_vectorized, y_train)
    print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_test, model.predict(vect.transform(X_test)))))

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def PlotWordCloud(words, title):
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white' 
                ).generate(words) 
                                                           
    # plot the WordCloud image                        
    plt.figure(figsize = (10, 10), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.title(title, fontsize=50)

    plt.show() 
negative = ''
for word in feature_names[sorted_coef_index[:100]]:
    negative += word + ' '
PlotWordCloud(negative, 'Most negative words')

positive = ''
for word in feature_names[sorted_coef_index[:-101:-1]]:
    positive += word + ' '    
PlotWordCloud(positive, 'Most positive words')

print(model.predict(vect.transform(df.tweet[0:10])))

#df=pd.read_csv('final_sentiment.csv')

df['p_score']=model.predict(vect.transform(df.tweet[0:]))
df.to_csv('sentiment.csv', index=False, encoding="utf-8")

""" # extracting 1-grams and 2-grams
vect = TfidfVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

len(vect.get_feature_names())

for c in c_val:
    model = LogisticRegression(C=c,solver='saga')
    model.fit(X_train_vectorized, y_train)
    print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_test, model.predict(vect.transform(X_test)))))

feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

print(model.predict(vect.transform(df.tweet[0:10])))
 """