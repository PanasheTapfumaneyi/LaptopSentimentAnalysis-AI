import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

# === 1. Load Dataset ===
df = pd.read_csv("cleaned_laptops_dataset.csv")
df.dropna(subset=["cleaned_review", "rating"], inplace=True)

def label_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

df["sentiment"] = df["rating"].apply(label_sentiment)

# === 2. Load GloVe Embeddings ===
def load_glove_embeddings(glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector
    return embeddings_index

glove_path = "glove.6B.100d.txt"  # You can change to 50d, 200d etc.
embedding_dim = 100
embeddings_index = load_glove_embeddings(glove_path)

# === 3. Text Preprocessing and Vectorization ===
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text.split()

def review_to_vector(review, embeddings_index, embedding_dim=100):
    words = preprocess_text(review)
    valid_words = [embeddings_index[word] for word in words if word in embeddings_index]
    if valid_words:
        return np.mean(valid_words, axis=0)
    else:
        return np.zeros(embedding_dim)

# Convert all reviews to embedding vectors
X_vect = np.vstack([review_to_vector(text, embeddings_index, embedding_dim) for text in df["cleaned_review"]])
y = df["sentiment"]

# === 4. Balance Data with SMOTE ===
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_vect, y)

# === 5. Train/Test Split ===
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]

# === 6. Train Classifier ===
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", solver="saga", random_state=42)
params = {
    'C': [0.1, 1, 10]
}
grid = GridSearchCV(log_reg, param_grid=params, scoring='f1_macro', cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# === 7. Evaluation ===
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid.best_params_)
print("Classification Report:\n", classification_report(y_test, y_pred))

conf_mat = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"])
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='coolwarm', xticklabels=["positive", "neutral", "negative"], yticklabels=["positive", "neutral", "negative"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# === 8. Save Model ===
# Save both model and embeddings if needed
joblib.dump(best_model, "glove_sentiment_model.pkl")
print("✅ Sentiment model saved as 'glove_sentiment_model.pkl'")
