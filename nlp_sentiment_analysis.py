# NLP Sentiment Analysis Project
# Based on IMDB Reviews example

import re
import pandas as pd
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
reviews = [
    ("This movie was absolutely fantastic! Best film of the year.", "positive"),
    ("Terrible acting and boring plot. Complete waste of time!", "negative"),
    ("Loved every minute of it. The director did an amazing job.", "positive"),
    ("Awful storyline. I fell asleep halfway through.", "negative"),
    ("Brilliant performances! The cinematography was breathtaking.", "positive"),
    ("Worst movie I have ever seen. Do not waste your money.", "negative"),
]

df = pd.DataFrame(reviews, columns=["review", "sentiment"])

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["clean_review"] = df["review"].apply(preprocess)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_review"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

sample_review = ["The movie was amazing and very enjoyable"]
sample_clean = [preprocess(sample_review[0])]
sample_vector = vectorizer.transform(sample_clean)

print("Prediction:", model.predict(sample_vector)[0])
