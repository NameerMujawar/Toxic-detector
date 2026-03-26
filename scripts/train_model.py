import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                               recall_score, f1_score,
                               confusion_matrix, classification_report)
from preprocess import prepare_data  # our preprocessing functions
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load and preprocess data ──────────────────────────────────────────
X_train, X_test, y_train, y_test, _ = prepare_data('dataset/processed.csv')

# ── 2. Define all three models ───────────────────────────────────────────
models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight='balanced', C=1.0
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
    )
}
# class_weight='balanced' tells the model to pay more attention to the
# minority (toxic) class — fixes the class imbalance problem!

results = {}  # store metrics for comparison

# ── 3. Train, evaluate, and compare each model ───────────────────────────
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"  Training: {name}")
    print(f"{'='*50}")

    # Train the model on training data
    model.fit(X_train, y_train)

    # Make predictions on the unseen test data
    y_pred = model.predict(X_test)

    # Calculate all metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    results[name] = {'Accuracy': acc, 'Precision': prec,
                     'Recall': rec, 'F1': f1, 'model': model}

    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Non-Toxic', 'Toxic']))

    # Plot confusion matrix for this model
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Non-Toxic', 'Toxic'],
                yticklabels=['Non-Toxic', 'Toxic'])
    plt.title(f'Confusion Matrix — {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'model/confusion_{name.replace(" ","_")}.png', dpi=150)
    plt.close()

# ── 4. Pick the best model by F1 score ───────────────────────────────────
best_name = max(results, key=lambda x: results[x]['F1'])
best_model = results[best_name]['model']
print(f"\n🏆 Best model: {best_name} (F1 = {results[best_name]['F1']:.4f})")

# ── 5. Save the best model ────────────────────────────────────────────────
with open('model/toxic_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✅ Best model saved to model/toxic_model.pkl")

# ── 6. Summary table ──────────────────────────────────────────────────────
summary = pd.DataFrame({
    name: {k: v for k, v in data.items() if k != 'model'}
    for name, data in results.items()
}).T
print("\n📊 Model Comparison Summary:")
print(summary.round(4))