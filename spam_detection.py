import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# Read the CSV file with the 'latin-1' encoding
data = pd.read_csv("spam.csv", encoding='latin-1')

# Split the data into input features (X) and target variable (y)
X = data['v2']
y = data['v1']

# Convert text data into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save the model using pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

file_size = os.path.getsize('model.pkl')
print("Model file size:", file_size, "bytes")