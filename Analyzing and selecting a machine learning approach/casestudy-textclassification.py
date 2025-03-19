from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Data teks sederhana (simulasi email)
emails = [
    "Beli sekarang, dapatkan diskon besar!", 
    "Dapatkan hadiah gratis hanya hari ini", 
    "Meeting jam 3 sore di kantor", 
    "Penawaran menarik dari Dicoding",
    "audit laporan keuangan bulan ini telah dikirim", 
    "Diskusi - saat bulan puasa, Mulut menjadi lebih bau",
    "Rekomendasi - lihat rekomendasi pakaian lebaran terb...",
    "Promo terbatas hanya untuk Anda"
]
labels = [1, 1, 0, 1, 0, 1, 1, 1]  # 1 = spam, 0 = tidak spam

# Pipeline = Vectorizer + Model
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()), 
    ('classifier', MultinomialNB())
])

# Grid Search dengan parameter yang sederhana
param_grid = {'classifier__alpha': [0.1, 0.5, 1.0]}
grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy')
grid_search.fit(emails, labels)

# Output hasil terbaik
print(f"Best Model: {grid_search.best_params_}, Accuracy: {grid_search.best_score_:.4f}")
