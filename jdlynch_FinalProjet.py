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
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from scipy.sparse import hstack
from scipy.sparse import csr_matrix

sia = SentimentIntensityAnalyzer()

nltk.download('stopwords')

def getScores(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['pos'], sentiment['neu'], sentiment['neg']

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def processText(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords])
    return text

df = pd.read_csv("train.csv", delimiter=",")
df = df.sample(frac=0.05)
stopwords = set(stopwords.words('english'))

X = df['text']  # Raw text reviews
y = df['polarity']        # Sentiment column: 1 for positive, 2 for negative

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_vader = np.array([getScores(text) for text in X_train])
X_test_vader = np.array([getScores(text) for text in X_test])

X_train_cleaned = X_train.apply(processText)
X_test_cleaned = X_test.apply(processText)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train_cleaned)
X_test_vec = vectorizer.transform(X_test_cleaned)

X_train_combined = hstack([X_train_vec, X_train_vader])
X_test_combined = hstack([X_test_vec, X_test_vader])

classifierKNN = KNeighborsClassifier(n_neighbors=3)
classifierKNN.fit(X_train_combined, y_train)
otherClassifierTestPred = classifierKNN.predict(X_test_combined)
npYtest = np.array(y_test)
print("K-Nearest Neighbour " + " Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

classifierRndForest = RandomForestClassifier(verbose=True)
classifierRndForest.fit(X_train_combined, y_train)
otherClassifierTestPred = classifierRndForest.predict(X_test_combined)
npYtest = np.array(y_test)
print("Random Forest " + " Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

X_train_combined_dense = csr_matrix(X_train_combined).toarray()
X_test_combined_dense = csr_matrix(X_test_combined).toarray()

classifierNB = GaussianNB()
classifierNB.fit(X_train_combined_dense, y_train)
otherClassifierTestPred = classifierNB.predict(X_test_combined_dense)
npYtest = np.array(y_test)
print("Gaussian NB" + " Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

classifierSVM = svm.LinearSVC()
classifierSVM.fit(X_train_combined, y_train)
otherClassifierTestPred = classifierSVM.predict(X_test_combined)
npYtest = np.array(y_test)
print("SVM " + " Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train_vec, y_train)
Y_pred = clf.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, Y_pred))
print("\nClassification Report:\n", classification_report(y_test, Y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, Y_pred))
