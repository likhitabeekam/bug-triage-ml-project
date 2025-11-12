import joblib
import pandas as pd

# Load trained models and vectorizers
cat_model = joblib.load("category_model.pkl")
cat_vectorizer = joblib.load("category_vectorizer.pkl")

pri_model = joblib.load("priority_model.pkl")
pri_vectorizer = joblib.load("priority_vectorizer.pkl")

# User gives a new bug report
summary = input("Enter bug summary: ")
description = input("Enter bug description: ")

text = summary + " " + description

# ---- Predict Category ----
X_cat = cat_vectorizer.transform([text])
predicted_category = cat_model.predict(X_cat)[0]

# ---- Predict Priority ----
X_pri = pri_vectorizer.transform([text])
predicted_priority = pri_model.predict(X_pri)[0]

print("\nPredicted Category:", predicted_category)
print("Predicted Priority:", predicted_priority)
