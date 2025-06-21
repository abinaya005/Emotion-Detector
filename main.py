import pandas as pd

# Load train data
train_df = pd.read_csv('data/train.txt', names=['text', 'emotion'], sep=';')
test_df = pd.read_csv('data/test.txt', names=['text', 'emotion'], sep=';')

print(train_df.head())
from sklearn.feature_extraction.text import TfidfVectorizer

X_train = train_df['text']
y_train = train_df['emotion']

X_test = test_df['text']
y_test = test_df['emotion']

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate on test data
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
while True:
    user_input = input("Enter a sentence (or 'exit' to stop): ")
    if user_input.lower() == "exit":
        break
    input_vec = vectorizer.transform([user_input])
    emotion = model.predict(input_vec)[0]
    print("Detected Emotion:", emotion)
