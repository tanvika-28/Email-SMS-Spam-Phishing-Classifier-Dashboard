# train_model.py

# ================= STEP 0: Install Required Libraries =================
import sys
import subprocess
import importlib


def install_and_import(package):
    try:
        importlib.import_module(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


required_packages = ["pandas", "numpy", "nltk", "scikit-learn", "matplotlib", "seaborn", "wordcloud", "xgboost"]
for pkg in required_packages:
    install_and_import(pkg)

# ================= STEP 1: Imports =================
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, StackingClassifier, RandomForestClassifier
import pickle

# ================= STEP 2: NLTK Download =================
nltk.download('punkt')
nltk.download('stopwords')

# ================= STEP 3: Load Data =================
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode target
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Drop duplicates
df = df.drop_duplicates(keep='first')

# ================= STEP 4: Feature Engineering =================
# Add text stats
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# ================= STEP 5: Text Preprocessing =================
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()

    y = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = y[:]
    y.clear()

    y = [ps.stem(i) for i in text]

    return " ".join(y)


df['transformed_text'] = df['text'].apply(transform_text)

# ================= STEP 6: Vectorization =================
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# ================= STEP 7: Train Models =================
# Base models
svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

# Voting Classifier
voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)], voting='soft')
voting.fit(X_train, y_train)
y_pred_voting = voting.predict(X_test)
print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred_voting))
print("Voting Classifier Precision:", precision_score(y_test, y_pred_voting))

# Stacking Classifier
estimators = [('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator = RandomForestClassifier()
stacking = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
stacking.fit(X_train, y_train)
y_pred_stack = stacking.predict(X_test)
print("Stacking Classifier Accuracy:", accuracy_score(y_test, y_pred_stack))
print("Stacking Classifier Precision:", precision_score(y_test, y_pred_stack))

# ================= STEP 8: Save Model and Vectorizer =================
# Using MultinomialNB as final model
mnb.fit(X_train, y_train)
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))

print("Model and vectorizer saved successfully!")
