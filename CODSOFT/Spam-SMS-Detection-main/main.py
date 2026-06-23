import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv("spam.csv", encoding="latin-1")
print(df.head())
df = df[["v1", "v2"]]
df.columns = ["label", "message"]
print(df.head())
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})
print(df.head())
X = df["message"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")
while True:
    msg = input("\nEnter SMS: ")
    if msg.lower() == "exit":
        break

    msg_vector = vectorizer.transform([msg])

    prediction = model.predict(msg_vector)

    if prediction[0] == 1:
        print("Spam Message")
    else:
        print("Legitimate Message")