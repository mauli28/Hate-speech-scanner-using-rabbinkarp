import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Sample dataset (you can expand later)
data = {
    "text": [
        "I hate you",
        "You are stupid",
        "I will kill you",
        "You are amazing",
        "Great job",
        "I love this",
        "You are dumb",
        "This is awesome"
    ],
    "label": [1,1,1,0,0,0,1,0]  # 1 = toxic, 0 = non-toxic
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

# Train model
model = LogisticRegression()
model.fit(X, df["label"])

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved!")
