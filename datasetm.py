import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Memuat dataset
df = pd.read_csv('students.csv', encoding='utf-8')

# Menampilkan informasi dasar dataset
print(f"Ukuran dataset: {df.shape}")
print(df.dtypes)
print(df.isnull().sum())
print(df.describe(include='all'))

# Konversi kolom 'Usia' ke numerik dan isi nilai yang hilang dengan median
df['Usia'] = pd.to_numeric(df['Usia'], errors='coerce')
df['Usia'].fillna(df['Usia'].median(), inplace=True)

# Visualisasi distribusi usia
plt.figure(figsize=(8, 6))
sns.histplot(df['Usia'], bins=5, kde=True)
plt.title('Distribusi Usia Mahasiswa')
plt.xlabel('Usia')
plt.ylabel('Frekuensi')
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# Distribusi Nilai: Visualisasikan distribusi nilai untuk setiap mata pelajaran. 
# Boxplot untuk nilai setiap mata pelajaran 
plt.figure(figsize=(10, 6)) 
sns.boxplot(data=df[['Matematika', 'IPA', 'Bahasa Inggris']]) 
plt.title('Distribusi Nilai Mahasiswa') 
plt.xlabel('Mata Pelajaran') 
plt.ylabel('Nilai') 
plt.show() 

 
# Perbandingan Nilai Berdasarkan Jenis Kelamin dan Jurusan 
# Visualisasikan perbandingan nilai berdasarkan jenis kelamin dan jurusan. 
# Scatter plot untuk perbandingan nilai berdasarkan jenis kelamin 
plt.figure(figsize=(10, 6)) 
sns.scatterplot(x='Matematika', y='IPA', hue='Jenis Kelamin', style='Jurusan', data=df, s=100) 
plt.title('Perbandingan Nilai Matematika dan IPA Berdasarkan Jenis Kelamin dan Jurusan') 
plt.xlabel('Nilai Matematika') 
plt.ylabel('Nilai IPA') 
plt.legend(title='Jenis Kelamin & Jurusan') 
plt.show() 
 
# Boxplot untuk perbandingan nilai berdasarkan jurusan 
plt.figure(figsize=(10, 6)) 
sns.boxplot(x='Jurusan', y='Matematika', hue='Jenis Kelamin', data=df) 
plt.title('Perbandingan Nilai Matematika Berdasarkan Jurusan dan Jenis Kelamin') 
plt.xlabel('Jurusan') 
plt.ylabel('Nilai Matematika') 
plt.show()