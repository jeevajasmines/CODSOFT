import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train = pd.read_csv("fraudTrain.csv")
test = pd.read_csv("fraudTest.csv")

print("Original Class Distribution:")
print(train['is_fraud'].value_counts())

fraud = train[train['is_fraud'] == 1]
legit = train[train['is_fraud'] == 0]

legit_sample = legit.sample(n=len(fraud), random_state=42)

train = pd.concat([fraud, legit_sample])
train = train.sample(frac=1, random_state=42)

print("\nBalanced Class Distribution:")
print(train['is_fraud'].value_counts())

train['trans_date_trans_time'] = pd.to_datetime(train['trans_date_trans_time'])
test['trans_date_trans_time'] = pd.to_datetime(test['trans_date_trans_time'])

train['hour'] = train['trans_date_trans_time'].dt.hour
test['hour'] = test['trans_date_trans_time'].dt.hour

cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']
train = pd.get_dummies(train, columns=cols)
test = pd.get_dummies(test, columns=cols)

train, test = train.align(test, join='left', axis=1, fill_value=0)
drop_cols = [
    'Unnamed: 0',
    'trans_date_trans_time',
    'cc_num',
    'first',
    'last',
    'street',
    'trans_num',
    'dob',
    'is_fraud'
]

X_train = train.drop(columns=drop_cols, errors='ignore')
y_train = train['is_fraud']


X_test = test.drop(columns=drop_cols, errors='ignore')
y_test = test['is_fraud']
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy :", accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nSample Prediction:")

if y_pred[0] == 1:
    print("Fraud Transaction")
else:
    print("Legitimate Transaction")