import joblib
import numpy as np
import re
import string

# === Load model ===
model = joblib.load("glove_sentiment_model_v2.pkl")

# === Load GloVe embeddings ===
def load_glove_embeddings(glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector
    return embeddings_index

glove_path = "glove.6B.100d.txt"  # Must match training time
embedding_dim = 100
embeddings_index = load_glove_embeddings(glove_path)

# === Preprocessing and vectorizing ===
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

# === Predict function ===
def predict_sentiment(review_text):
    review_vector = review_to_vector(review_text, embeddings_index, embedding_dim).reshape(1, -1)
    prediction = model.predict(review_vector)[0]
    return prediction

# === Example usage ===
review = "It is okay but good"
predicted_sentiment = predict_sentiment(review)
print(f"Predicted Sentiment: {predicted_sentiment}")
