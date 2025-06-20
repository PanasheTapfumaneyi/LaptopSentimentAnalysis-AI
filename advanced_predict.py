import joblib
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data
nltk.download(['stopwords', 'wordnet', 'punkt'], quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = string.punctuation + '‘’“”–—'
        
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and token not in self.punctuation]
        return tokens


# Load model with metadata
model_data = joblib.load("glove_sentiment_model_v2.pkl")
model = model_data['model']
preprocessor = model_data.get('preprocessor', None)
embedding_dim = model_data['embeddings_info']['dim']

# Load GloVe embeddings
def load_glove_embeddings(glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector
    return embeddings_index

glove_path = "glove.6B.100d.txt"  
embeddings_index = load_glove_embeddings(glove_path)

# Text Preprocessing
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = string.punctuation + '‘’“”–—'
        
    def preprocess(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, special chars, numbers
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and punctuation, lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and token not in self.punctuation]
        
        return tokens

# Use the saved preprocessor if available, otherwise create new
if preprocessor is None:
    preprocessor = TextPreprocessor()

# === Vectorization ===
def review_to_vector(review, embeddings_index, embedding_dim):
    words = preprocessor.preprocess(review)
    vectors = [embeddings_index[word] for word in words if word in embeddings_index]
    
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(embedding_dim)

# === Prediction with confidence scores ===
def predict_sentiment(review_text):
    # Vectorize the review
    review_vector = review_to_vector(review_text, embeddings_index, embedding_dim).reshape(1, -1)
    
    # Get prediction and probabilities
    prediction = model.predict(review_vector)[0]
    probabilities = model.predict_proba(review_vector)[0]
    
    # Map to class labels
    class_mapping = {i: cls for i, cls in enumerate(model.classes_)}
    confidence_scores = {class_mapping[i]: float(prob) 
                        for i, prob in enumerate(probabilities)}
    
    return {
        'prediction': prediction,
        'confidence': confidence_scores,
        'confidence_score': float(probabilities.max())
    }

if __name__ == "__main__":
    test_reviews = [
        "This laptop is absolutely fantastic! The speed is incredible.",
        "The battery life is terrible and it overheats constantly.",
        "It's okay, nothing special but gets the job done.",
        "The screen quality is average but the keyboard feels cheap."
    ]
    
    for review in test_reviews:
        result = predict_sentiment(review)
        print(f"\nReview: {review}")
        print(f"Predicted sentiment: {result['prediction']}")
        print(f"Confidence scores: {result['confidence']}")
        print(f"Highest confidence: {result['confidence_score']:.2%}")