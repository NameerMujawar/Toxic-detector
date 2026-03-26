import pickle

# ─────────────────────────────────────────────────────────────────────────
# SAVING a model (done after training)
# ─────────────────────────────────────────────────────────────────────────

def save_model(model, filepath):
    """Save a trained model to disk."""
    with open(filepath, 'wb') as f:   # 'wb' = write binary
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load a saved model from disk."""
    with open(filepath, 'rb') as f:   # 'rb' = read binary
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model

# ─────────────────────────────────────────────────────────────────────────
# LOADING and running a prediction
# ─────────────────────────────────────────────────────────────────────────

model = load_model('model/toxic_model.pkl')
vectorizer = load_model('model/tfidf_vectorizer.pkl')

def predict_toxicity(message):
    """
    Takes a raw message string and returns:
      - label: 'toxic' or 'safe'
      - score: probability as 0-100% integer
    """
    from preprocess import clean_text

    cleaned = clean_text(message)             # clean the text
    vector  = vectorizer.transform([cleaned]) # convert to TF-IDF vector
    proba   = model.predict_proba(vector)[0]  # [prob_safe, prob_toxic]
    score   = int(proba[1] * 100)             # toxic probability as 0-100
    label   = "toxic" if score >= 50 else "safe"

    return label, score

# Testing
for msg in ["Hello everyone, hope you have a great day!",
             "I hate you, you're the worst person alive"]:
    label, score = predict_toxicity(msg)
    print(f"Message: {msg[:50]}")
    print(f"  → {label.upper()} ({score}% toxic)\n")