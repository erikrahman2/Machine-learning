from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np

# Load dataset contoh (Iris)
data = load_iris()
X, y = data.data, data.target  # Fitur dan label

# Contoh K-Fold Cross-Validation
model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# Cetak hasil akurasi
print(f"Accuracy: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")
