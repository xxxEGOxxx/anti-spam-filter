import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

text = "text"
target = "target"

df = pd.read_csv("trainingSet.csv", encoding="utf-8").rename(columns={"v1": "target", "v2": "text"})[["text", "target"]]

size = df[target].size

print("all\t\t" + str(size))

counts = df[target].value_counts()

print(counts)

X = df["text"].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

# naive_bayes
naive_bayes = make_pipeline(CountVectorizer(binary=True), MultinomialNB())

naive_bayes.fit(X_train, y_train)
y_test_pred = naive_bayes.predict(X_test)

print(classification_report(y_test, y_test_pred))

# logical_regression
logical_regression = make_pipeline(CountVectorizer(binary=True), LogisticRegression())

logical_regression.fit(X_train, y_train)
y_test_pred = logical_regression.predict(X_test)

print(classification_report(y_test, y_test_pred))
