# Install required libraries if not already installed
# !pip install pandas scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset
# Sample dataset: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
df = pd.read_csv('fake_or_real_news.csv')  # Replace with your file path

# 2. Prepare Data
X = df['text']  # News content
y = df['label']  # Labels: 'FAKE' or 'REAL'

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Text Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 6. Make Predictions
y_pred = model.predict(X_test_vec)

# 7. Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Try with new text
sample_news = ["The government passed a new healthcare bill today.", 
               "NASA confirmed alien life on Mars!"]
sample_vec = vectorizer.transform(sample_news)
print("Predictions:", model.predict(sample_vec))
