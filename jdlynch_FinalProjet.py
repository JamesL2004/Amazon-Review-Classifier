import numpy as np
import pandas as pd
import nltk
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

def processText(text):

    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    text = ' '.join([word for word in text.split() if word not in stopwords])

    return text

df = pd.read_csv("train.csv", delimiter=",")
stopwords = set(stopwords.words('english'))

X = df['text']  # Raw text reviews
y = df['polarity']        # Sentiment column: 1 for positive, 2 for negative

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_cleaned = X_train.apply(processText)
X_test_cleaned = X_test.apply(processText)

vectorizer = TfidfVectorizer(max_features=5000)  # Keep top 5000 features
X_train_vec = vectorizer.fit_transform(X_train_cleaned)
X_test_vec = vectorizer.transform(X_test_cleaned)

classifierKNN = KNeighborsClassifier(n_neighbors=3)
classifierKNN.fit(X_train_vec, y_train)
otherClassifierTestPred = classifierKNN.predict(X_test_vec)
npYtest = np.array(y_test)
print("K-Nearest Neighbour " + " Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

classifierRndForest = RandomForestClassifier(verbose=True)
classifierRndForest.fit(X_train_vec, y_train)
otherClassifierTestPred = classifierRndForest.predict(X_test_vec)
npYtest = np.array(y_test)
print("Random Forest " + " Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

classifierNB = GaussianNB()
classifierNB.fit(X_train_vec.toarray(), y_train)
otherClassifierTestPred = classifierNB.predict(X_test_vec.toarray())
npYtest = np.array(y_test)
print("Gaussian NB" + " Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

classifierSVM = svm.LinearSVC()
classifierSVM.fit(X_train_vec, y_train)
otherClassifierTestPred = classifierSVM.predict(X_test_vec)
npYtest = np.array(y_test)
print("SVM " + " Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))