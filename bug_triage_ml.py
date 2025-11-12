import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
df = pd.read_csv('sample_bugs_dataset.csv')

# Combine title and description for text input
df['text'] = df['summary'] + ' ' + df['description']

# --- 1. CATEGORY CLASSIFICATION ---
X = df['text']
y_category = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y_category, test_size=0.2, random_state=42)

vectorizer_cat = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X_train_tfidf = vectorizer_cat.fit_transform(X_train)
X_test_tfidf = vectorizer_cat.transform(X_test)

model_category = LogisticRegression(max_iter=1000)
model_category.fit(X_train_tfidf, y_train)
y_pred_cat = model_category.predict(X_test_tfidf)

print("Category Classification Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_cat))
print("Precision:", precision_score(y_test, y_pred_cat, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_cat, average='weighted'))
print("F1-Score:", f1_score(y_test, y_pred_cat, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_cat))

# --- 2. PRIORITY CLASSIFICATION ---
y_priority = df['priority']
X_train, X_test, y_train, y_test = train_test_split(X, y_priority, test_size=0.2, random_state=42)

vectorizer_pri = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X_train_tfidf = vectorizer_pri.fit_transform(X_train)
X_test_tfidf = vectorizer_pri.transform(X_test)

model_priority = LogisticRegression(max_iter=1000)
model_priority.fit(X_train_tfidf, y_train)
y_pred_pri = model_priority.predict(X_test_tfidf)

print("\nPriority Prediction Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_pri))
print("Precision:", precision_score(y_test, y_pred_pri, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_pri, average='weighted'))
print("F1-Score:", f1_score(y_test, y_pred_pri, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_pri))


# ---------------- ADD GRAPH CODE HERE ----------------
# import matplotlib.pyplot as plt
# import seaborn as sns

# # PLOT RESULTS FOR CATEGORY CLASSIFICATION
# category_metrics = {
#     'Accuracy': accuracy_score(y_test, y_pred_cat),
#     'Precision': precision_score(y_test, y_pred_cat, average='weighted'),
#     'Recall': recall_score(y_test, y_pred_cat, average='weighted'),
#     'F1-Score': f1_score(y_test, y_pred_cat, average='weighted')
# }

# plt.figure(figsize=(6, 4))
# plt.bar(category_metrics.keys(), category_metrics.values())
# plt.title("Category Classification Metrics")
# plt.ylabel("Score")
# plt.ylim(0, 1)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

# plt.figure(figsize=(6, 5))
# sns.heatmap(confusion_matrix(y_test, y_pred_cat), annot=True, cmap='Blues', fmt='d')
# plt.title("Category Classification Confusion Matrix")
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.show()


# # PLOT RESULTS FOR PRIORITY CLASSIFICATION
# priority_metrics = {
#     'Accuracy': accuracy_score(y_test, y_pred_pri),
#     'Precision': precision_score(y_test, y_pred_pri, average='weighted'),
#     'Recall': recall_score(y_test, y_pred_pri, average='weighted'),
#     'F1-Score': f1_score(y_test, y_pred_pri, average='weighted')
# }

# plt.figure(figsize=(6, 4))
# plt.bar(priority_metrics.keys(), priority_metrics.values(), color='orange')
# plt.title("Priority Classification Metrics")
# plt.ylabel("Score")
# plt.ylim(0, 1)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

# plt.figure(figsize=(6, 5))
# sns.heatmap(confusion_matrix(y_test, y_pred_pri), annot=True, cmap='Oranges', fmt='d')
# plt.title("Priority Classification Confusion Matrix")
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.show()

# --- 3. DUPLICATE DETECTION ---
vectorizer_dup = TfidfVectorizer(max_features=2000, stop_words='english')
text_features = vectorizer_dup.fit_transform(df['text'])

cos_sim = cosine_similarity(text_features)
dup_pairs = []
threshold = 0.85

for i in range(len(df)):
    for j in range(i+1, len(df)):
        if cos_sim[i][j] > threshold:
            dup_pairs.append((df.loc[i, 'bug_id'], df.loc[j, 'bug_id'], cos_sim[i][j]))


print("\n" + "="*60)
print("          DUPLICATE DETECTION RESULTS")
print("="*60)
print(f"Similarity Threshold: {threshold}")
print(f"Total Similar Pairs Found: {len(dup_pairs)}\n")

# Display first few sample pairs neatly
print("Top 5 Sample Duplicate Pairs (Bug1 → Bug2 | Similarity Score):")
print("-" * 60)
for idx, (bug1, bug2, score) in enumerate(dup_pairs[:5], 1):
    print(f"{idx}. Bug {int(bug1)}  →  Bug {int(bug2)}  |  Similarity: {score:.3f}")
print("-" * 60)

if len(dup_pairs) == 0:
    print("No duplicate pairs found above the given threshold.")
else:
    print(f"\n✅ Duplicate detection completed successfully.\n")



# --- 4. SAVE MODELS AND VECTORIZERS ---
import joblib

# Save trained models and vectorizers
joblib.dump(model_category, "category_model.pkl")
joblib.dump(vectorizer_cat, "category_vectorizer.pkl")

joblib.dump(model_priority, "priority_model.pkl")
joblib.dump(vectorizer_pri, "priority_vectorizer.pkl")

print("✅ Models saved successfully.")
