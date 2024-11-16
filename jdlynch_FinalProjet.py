import numpy as np
import pandas as pd
import nltk
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

df = pd.read_csv("train.csv", delimiter=",")
stopwords = set(stopwords.words('english'))

def processText(text):

    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    text = ' '.join([word for word in text.split() if word not in stopwords])

    return text

df['cleaned_text'] = df['text'].head(50).apply(processText)

print(df['cleaned_text'].head(5))