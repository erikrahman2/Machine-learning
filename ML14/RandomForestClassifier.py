import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Contoh data latih
X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # Fitur
y_train = np.array([0, 1, 0, 1])  # Label

# Membuat dan melatih model RandomForest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Menyimpan model ke file model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model sudah disimpan sebagai model.pkl")
