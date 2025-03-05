import pandas as pd
import numpy as np  # Pastikan numpy di-import untuk menangani np.nan

# Membaca data CSV dengan memastikan 'np.nan' dikenali sebagai NaN
df = pd.read_csv('data.csv', na_values=['', 'np.nan'])

# Menampilkan jumlah missing values
print("Missing Values per Kolom:")
print(df.isnull().sum())

# Mengisi Missing Values pada kolom 'Age'
df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].mean())   # Menggunakan mean
# df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].median())  # Menggunakan median
# df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].mode()[0]) # Menggunakan mode

# Menampilkan data setelah perbaikan
print("\nData setelah penanganan missing values:")
print(df)

# Identifikasi duplikat sebelum penghapusan
duplicates_count = df.duplicated().sum()
print(f"\nJumlah data duplikat sebelum dihapus: {duplicates_count}")

# Menghapus duplikat
df.drop_duplicates(inplace=True)

# Verifikasi setelah penghapusan
print(f"Jumlah data setelah penghapusan duplikat: {df.shape[0]} baris")