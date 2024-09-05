import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, KFold

df = pd.read_csv('/content/url_data.csv', header=None, names=['index', 'url', 'category'])

df = df.drop(columns=['index'])

# Handle NaN values by removing them
df = df.dropna()

# Display the first few rows to verify
print(df.head())

vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split('/'), stop_words='english')
X = vectorizer.fit_transform(df['url'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['category'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(nb_classifier, X, y, cv=kf, scoring='accuracy')

print(f'Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Mean Cross-Validation Accuracy: {cv_scores.mean()}')
