import pickle
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.preprocess import clean_text

class ToxicityPredictor:
    """Loads the saved model and vectorizer, exposes a predict method."""

    def __init__(self):
        print("Loading ML model and vectorizer...")
        with open('model/toxic_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open('model/tfidf_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        print("✅ Model loaded and ready.")

    def predict(self, message: str) -> dict:
        """
        Returns a dict:
          { 'is_toxic': bool, 'score': int (0-100), 'label': str }
        """
        # 🔴 Indian toxic word override
        INDIAN_TOXIC_WORDS = [
        "chutiya", "bkl", "mc", "bc", "madarchod",
        "behenchod", "gandu", "harami", "kutte",
        "lavde", "lund", "randi","hijde","neech","gand","aand","zhaatu",
        "zhaat","lode","bhosdike","madarchod","behen ke lode","chodu",
        "gaand","bhenchod"]

        msg_lower = message.lower()

        for word in INDIAN_TOXIC_WORDS:
            if word in msg_lower:
                return {
                    "is_toxic": True,
                    "score": 95,
                    "label": "TOXIC"
                }
        
        # ── ML prediction ─────────────────────────────
        cleaned  = clean_text(message)
        vector   = self.vectorizer.transform([cleaned])
        proba    = self.model.predict_proba(vector)[0]
        score    = int(proba[1] * 100)  # index 1 = toxic class probability
        is_toxic = score >= 50

        return {
            "is_toxic": is_toxic,
            "score": score,
            "label": "TOXIC" if is_toxic else "SAFE"
        }