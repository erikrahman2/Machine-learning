from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np

# Load dataset Iris
data = load_iris()
X, y = data.data, data.target  # Fitur dan label

# Membagi dataset menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definisikan model dan parameter distribusi
model = RandomForestClassifier()
param_dist = {
    'n_estimators': [50, 100, 200],  # Jumlah pohon dalam hutan
    'max_depth': [None, 10, 20, 30],  # Kedalaman maksimum pohon
    'bootstrap': [True, False]  # Menggunakan bootstrap atau tidak
}

# Randomized Search dengan 10 iterasi dan 5-Fold Cross Validation
random_search = RandomizedSearchCV(
    estimator=model, param_distributions=param_dist, n_iter=10, cv=5, 
    scoring='accuracy', n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)

# Menampilkan hasil terbaik
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)
