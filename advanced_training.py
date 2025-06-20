import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data
nltk.download(['stopwords', 'wordnet', 'punkt', 'punkt_tab'])


# Dataset Cleaning 
def clean_dataset(df):
    # Validate if columns exist
    required_columns = ['cleaned_review', 'rating']
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    df = df.copy()
    
    # Remove commas and other symbols
    df['cleaned_review'] = df['cleaned_review'].astype(str).str.replace(r'[,\\\'"]', '', regex=True)
    
    # Fill missing values
    df['cleaned_review'] = df['cleaned_review'].fillna('')
    rating_median = df['rating'].median()
    df['rating'] = df['rating'].fillna(rating_median).astype(int)
    
    # Remove duplicates with fuzzy matching
    df = df.drop_duplicates(subset=['cleaned_review', 'rating'])
    
    # Remove any remaining null reviews
    df = df[df['cleaned_review'].str.strip().astype(bool)]
    
    return df.reset_index(drop=True)

# Load and clean dataset
try:
    df = pd.read_csv("cleaned_laptops_dataset.csv", encoding='utf-8')
    df = clean_dataset(df)
except Exception as error:
    print(f"Error occured while loading or cleaning dataset: {str(error)}")
    raise

# Sentiment Labeling
def label_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    elif rating >= 4:
        return "positive"
    else:
        raise ValueError(f"Invalid rating: {rating}")

df["sentiment"] = df["rating"].apply(label_sentiment)

# Check class distribution
print("\nClass Distribution:")
print(df["sentiment"].value_counts(normalize=True))

# Text Preprocessing
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = string.punctuation + '-_""'
        
    def preprocess(self, text):
        # Convert to lowercase
        text = text.lower()
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and punctuation
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and token not in self.punctuation]
        
        return tokens

preprocessor = TextPreprocessor()

# Print sample cleaned reviews before embedding
print("\nSample reviews before GloVe embedding:")
for i, review in enumerate(df['cleaned_review'][:3]):
    tokens = preprocessor.preprocess(review)
    print(f"Review {i+1}:")
    print("Original:", review)
    print("Tokens  :", tokens)
    print()

# GloVe Embeddings
def glove_embedding(glove_file):
    embeddings_index = {}
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index
    
glove_path = "glove.6B.100d.txt"
embeddings_index = glove_embedding(glove_path)
embedding_dimensions = 100


# Convert reviews to fixed size vector
def review_vectorization(review, embeddings_index, embedding_dimensions):
    # Preprocess words in review column 
    words = preprocessor.preprocess(review)
    # Embed if the words are in the index
    vectors = [embeddings_index[word] for word in words if word in embeddings_index]
    
    # If there is a word with known embedding return average vector
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    # If there are no words with known embeddings return a vector of zeros
    else:
        return np.zeros(embedding_dimensions)

# Vectorize the reviews
print("\nVectorizing reviews: ")
X_vector = np.array([review_vectorization(text, embeddings_index, embedding_dimensions) 
                  for text in df['cleaned_review']])
y = df['sentiment']

# Print sample vectors after embedding
print("\nSample reviews after GloVe embedding:")
for i in range(3):
    print(f"Review vector {i+1}: {X_vector[i]}")

# Data Balancing with SMOTE
print("\nClass distribution before SMOTE balancing:")
print(pd.Series(y).value_counts())

smote = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors= 10)
X_resampled, y_resampled = smote.fit_resample(X_vector,y)

print("\nClass distribution after SMOTE balancing:")
print(pd.Series(y_resampled).value_counts())

# Train-Test Split
splitter = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=42)
for train_idx, test_idx in splitter.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]

# Model Training

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(
        max_iter=5000,
        class_weight='balanced',
        solver='saga',
        random_state=42,
        n_jobs=-1
    ))
])

params = [
    # Grid for L1 and L2 penalties (no l1_ratio)
    {
        'clf__penalty': ['l1', 'l2'],
        'clf__C': np.logspace(-3, 3, 7),
    },
    # Grid for ElasticNet
    {
        'clf__penalty': ['elasticnet'],
        'clf__C': np.logspace(-3, 3, 7),
        'clf__l1_ratio': [0.1, 0.5, 0.9],
    }
]

grid = GridSearchCV(
    pipeline,
    param_grid=params,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("\nTraining model: ")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Model Evaluation
def model_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    conf_mat = confusion_matrix(y_test, y_pred, 
                              labels=["positive", "neutral", "negative"])
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Greens',
               xticklabels=["positive", "neutral", "negative"],
               yticklabels=["positive", "neutral", "negative"])
    plt.title("Confusion Matrix", pad=15)
    plt.xlabel("Predicted Label", labelpad=5)
    plt.ylabel("True Label", labelpad=5)
    plt.tight_layout()
    plt.show()
    
    # Feature importance
    if hasattr(model.named_steps['clf'], 'coef_'):
        print("\nTop predictive words:")
        classes = model.named_steps['clf'].classes_
        for i, class_label in enumerate(classes):
            print(f"\n{class_label.upper()} class:")
            sorted_features = np.argsort(model.named_steps['clf'].coef_[i])
            print("Most positive:", sorted_features[-10:])
            print("Most negative:", sorted_features[:10])

model_evaluation(best_model, X_test, y_test)

# Saving model with joblib
joblib.dump({
    'model': best_model,
    'preprocessor': preprocessor,
    'embeddings_info': {'dim': embedding_dimensions, 'vocab_size': len(embeddings_index)}
}, "review_sentiment_model.pkl")

print("\n Model saved as 'review_sentiment_model.pkl'")