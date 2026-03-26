import os
os.makedirs("model", exist_ok=True)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Use a clean style
plt.style.use('dark_background')
COLORS = ['#7c3aed', '#06b6d4', '#10b981', '#f59e0b']

# ── Chart 1: Toxic vs Non-Toxic distribution ─────────────────────────────
def plot_distribution(df):
    counts = df['is_toxic'].value_counts().sort_index()
    labels = ['Non-Toxic', 'Toxic']
    colors = ['#10b981', '#ef4444']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bars = ax1.bar(labels, counts.values, color=colors, width=0.5)
    ax1.set_title('Message Class Distribution', fontsize=14, pad=15)
    ax1.set_ylabel('Number of Messages')
    for bar, count in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f'{count:,}', ha='center', fontsize=11)

    # Pie chart
    ax2.pie(counts.values, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class Balance', fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig('model/chart_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved: model/chart_distribution.png")
    plt.show()


# ── Chart 2: Model accuracy comparison ───────────────────────────────────
def plot_model_comparison(results: dict):
    """results = {'Model Name': {'Accuracy': 0.9, 'F1': 0.85, ...}}
    """
    models  = list(results.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        vals = [results[m][metric] for m in models]
        bars = ax.bar(x + i * width, vals, width, label=metric, color=COLORS[i % len(COLORS)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.2f}', ha='center', fontsize=8)

    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, pad=15)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model/chart_model_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: model/chart_model_comparison.png")
    plt.show()


# ── Chart 3: TF-IDF top words per class ──────────────────────────────────
def plot_top_words(vectorizer, model, n=15):
    if not hasattr(model, 'coef_'):
        print("Top-words chart only works with Logistic Regression")
        return

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    top_toxic = feature_names[np.argsort(coefs)[:-n-1:-1]]
    top_safe  = feature_names[np.argsort(coefs)[:n]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.barh(range(n), np.sort(coefs)[:-n-1:-1], color='#ef4444')
    ax1.set_yticks(range(n)); ax1.set_yticklabels(top_toxic)
    ax1.set_title('Top Toxic Indicator Words', fontsize=13)
    ax1.invert_yaxis()

    ax2.barh(range(n), np.sort(coefs)[:n], color='#10b981')
    ax2.set_yticks(range(n)); ax2.set_yticklabels(top_safe)
    ax2.set_title('Top Safe Indicator Words', fontsize=13)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig('model/chart_top_words.png', dpi=150, bbox_inches='tight')
    print("Saved: model/chart_top_words.png")
    plt.show()

import pickle
import pandas as pd

# Load dataset
df = pd.read_csv("dataset\processed.csv")

# Load trained model
with open("model/toxic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load TFIDF vectorizer
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

results = {
    "Logistic Regression": {
        "Accuracy": 0.9268,
        "Precision": 0.5978,
        "Recall": 0.8570,
        "F1": 0.7043
    },
    "Random Forest": {
        "Accuracy": 0.9437,
        "Precision": 0.8002,
        "Recall": 0.5951,
        "F1": 0.6826
    },
    "Naive Bayes": {
        "Accuracy": 0.9488,
        "Precision": 0.8871,
        "Recall": 0.5692,
        "F1": 0.6934
    }
}

if __name__ == "__main__":
    plot_distribution(df)
    plot_model_comparison(results)
    plot_top_words(vectorizer, model)