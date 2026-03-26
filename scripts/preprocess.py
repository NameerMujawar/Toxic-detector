import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# Load English stopwords (words like 'the', 'a', 'is' that add no meaning)
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
    Cleans a raw text message through multiple steps.
    Each step removes noise so the model focuses on meaningful words.
    """
    # Step 1: Convert to lowercase (so 'HATE' and 'hate' are treated as same)
    text = text.lower()

    # Step 2: Remove URLs (http://... or www...)
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 3: Remove HTML tags (like <br> or <b>)
    text = re.sub(r'<.*?>', '', text)

    # Step 4: Remove numbers
    text = re.sub(r'\d+', '', text)

    # Step 5: Remove punctuation (!, ?, ., etc.)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 6: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 7: Tokenize (split into individual words)
    tokens = word_tokenize(text)

    # Step 8: Remove stopwords (common words that add little meaning)
    tokens = [word for word in tokens if word not in STOPWORDS]

    # Step 9: Join tokens back into a single string
    return ' '.join(tokens)


def prepare_data(csv_path, max_features=10000):
    """
    Loads the dataset, cleans text, applies TF-IDF, and splits data.
    Returns X_train, X_test, y_train, y_test AND saves the vectorizer.
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    print("Cleaning text (this may take a minute)...")
    df['clean_text'] = df['comment_text'].apply(clean_text)

    # Features (X) = cleaned text, Labels (y) = is_toxic
    X = df['clean_text']
    y = df['is_toxic']

    # Split: 80% training, 20% testing (standard ML practice)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # stratify=y ensures both splits have the same toxic/non-toxic ratio

    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set:  {len(X_test)} samples")

    # TF-IDF Vectorization
    # max_features=10000 means we use only the top 10,000 most important words
    print("Applying TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    # ngram_range=(1,2) means we use single words AND word pairs (bigrams)
    # e.g. "kill you" is more toxic than "kill" and "you" separately!

    X_train_tfidf = vectorizer.fit_transform(X_train)  # learn vocab + transform
    X_test_tfidf = vectorizer.transform(X_test)          # transform ONLY (don't re-learn)

    # Save the vectorizer — we'll need it in the Discord bot later
    with open('model/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("TF-IDF vectorizer saved to model/tfidf_vectorizer.pkl")

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


# Quick test of text cleaning
if __name__ == "__main__":
    sample = "You are HORRIBLE!! Visit http://spam.com for more info!!"
    print("Original:", sample)
    print("Cleaned: ", clean_text(sample))