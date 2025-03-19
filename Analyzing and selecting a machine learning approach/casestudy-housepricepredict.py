import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Dataset kecil (simulasi harga rumah)
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Random Forest
rf = RandomForestRegressor(random_state=42)

# Random Search dengan parameter ringan
param_dist = {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10]}
random_search = RandomizedSearchCV(rf, param_dist, n_iter=5, cv=3, scoring='neg_mean_absolute_error', random_state=42)
random_search.fit(X_train, y_train)

# Evaluasi
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Best Random Forest Model: {random_search.best_params_}, MAE: {mae:.4f}")
