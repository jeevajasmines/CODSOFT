genres=[]
descriptions=[]
file=open("train_data.txt","r",encoding="utf-8")

for line in file:
    parts=line.split(" ::: ")
    if len(parts)>=4:
        genres.append(parts[2])
        descriptions.append(parts[3])
file.close()
print("Total Movies:",len(descriptions))
print("First Genre:",genres[0])
print("First Description:",descriptions[0][:100])

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_features=5000)
X=tfidf.fit_transform(descriptions)
print("TF-IDF Shape:",X.shape)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
model.fit(X,genres)
print("Model Training Completed!")

user_input=input("Enter Movie Description:")
sample=[user_input]
sample_tfidf=tfidf.transform(sample)
prediction=model.predict(sample_tfidf)
print("Predicted Genre:",prediction[0])

from sklearn.metrics import accuracy_score
predictions=model.predict(X)
accuracy=accuracy_score(genres,predictions)
print("Accuracy:",accuracy)