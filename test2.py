import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

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
print("\nSample reviews before processing:")
for i, review in enumerate(df['cleaned_review'][:3]):
    tokens = preprocessor.preprocess(review)
    print(f"Review {i+1}:")
    print("Original:", review)
    print("Tokens:", tokens)
    print()

# GloVe Embeddings
def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

glove_path = "glove.6B.100d.txt"
embeddings_index = load_glove_embeddings(glove_path)
embedding_dim = len(next(iter(embeddings_index.values())))  # Get embedding dimension

# Tokenize text and prepare sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_review'])
sequences = tokenizer.texts_to_sequences(df['cleaned_review'])

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Pad sequences
max_length = 100  # Max sequence length
X = pad_sequences(sequences, maxlen=max_length)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentiment'])
y = to_categorical(y)  # Convert to one-hot encoding

# Create embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Before and after embedding
print("\n=== Text Before and After Embedding ===")

sample_reviews = df['cleaned_review'].head(2).tolist()

for i, review in enumerate(sample_reviews):
    print(f"\nSample {i+1}:")
    print("1. Original Text:", review)
    
    # After preprocessing
    tokens = preprocessor.preprocess(review)
    print("2. After Preprocessing:", tokens)
    
    # After tokenizer
    seq = tokenizer.texts_to_sequences([review])[0]
    print("3. Numerical Sequence:", seq)
    
    # After padding
    padded = X[i]
    print("4. Padded Sequence:", padded)
    print("   Length:", len(padded), "(padded to max_length)")

# Print sample word embedding
print("\n Sample Word Embedding")
test_word = "great"
if test_word in tokenizer.word_index:
    word_idx = tokenizer.word_index[test_word]
    print(f"Word: '{test_word}'")
    print(f"Index in vocabulary: {word_idx}")
    print(f"Embedding vector (first 5 values): {embedding_matrix[word_idx][:5]}...")
else:
    print(f"'{test_word}' not in vocabulary")

# Data Balancing with SMOTE
print("\nClass distribution before SMOTE balancing:")
print(df["sentiment"].value_counts())

# Flatten to X
X_flat = X.reshape(X.shape[0], -1)
smote = SMOTE(random_state=42)
X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y)

# Print class distribution after SMOTE
print("\nClass distribution after SMOTE balancing:")
y_resampled_classes = np.argmax(y_resampled, axis=1)
unique, counts = np.unique(y_resampled_classes, return_counts=True)
for class_idx, count in zip(unique, counts):
    class_name = label_encoder.inverse_transform([class_idx])[0]
    print(f"{class_name}: {count} samples")

# Reshape to original dimensions
X_resampled = X_resampled_flat.reshape(X_resampled_flat.shape[0], max_length)

# Convert y back to categorical 
y_resampled_categorical = np.argmax(y_resampled, axis=1)

# Train-Test Split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(X_resampled, y_resampled_categorical):
    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]

# Build RNN Model with GloVe embeddings
def build_rnn_model():
    model = Sequential()
    
    # Embedding layer with GloVe weights
    model.add(Embedding(input_dim=vocab_size, 
                       output_dim=embedding_dim, 
                       weights=[embedding_matrix],
                       input_length=max_length,
                       trainable=True))
    
    # Bidirectional RNN layer
    model.add(Bidirectional(SimpleRNN(128, return_sequences=True, 
                                   kernel_regularizer=l2(0.01))))
    model.add(Dropout(0.3))
    
    # Second RNN layer
    model.add(SimpleRNN(64, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    
    # Dense layer with regularization
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(y_train.shape[1], activation='softmax'))
    
    # Optimizer with learning rate schedule
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy', 
                         tf.keras.metrics.Precision(name='precision'),
                         tf.keras.metrics.Recall(name='recall')])
    return model

# Callbacks for better training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)

# Build and train model
model = build_rnn_model()
print("\nModel summary:")
model.summary()

print("\nTraining model...")
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=64,
                    validation_split=0.2,
                    verbose=1)

# Model Evaluation
def model_evaluation(model, X_test, y_test):
    # Predict classes
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    class_names = label_encoder.classes_
    
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                               target_names=class_names, digits=4))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Greens',
               xticklabels=class_names,
               yticklabels=class_names)
    plt.title("Confusion Matrix", pad=15)
    plt.xlabel("Predicted Label", labelpad=5)
    plt.ylabel("True Label", labelpad=5)
    plt.tight_layout()
    plt.show()
    
    # Training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

model_evaluation(model, X_test, y_test)

# Save model
joblib.dump({
    'model': model,
    'preprocessor': preprocessor,
    'tokenizer': tokenizer,
    'label_encoder': label_encoder,
    'max_len': max_length,
    'embedding_matrix': embedding_matrix
}, "rnn_glove_sentiment_model.pkl")

print("\nModel saved as 'rnn_glove_sentiment_model.pkl'")