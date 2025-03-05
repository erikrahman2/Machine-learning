import pandas as pd

# Membaca data CSV dan menangani missing values otomatis
df = pd.read_csv('data.csv', na_values=['', 'np.nan'])

# Menampilkan jumlah missing values sebelum perbaikan
print("Missing Values per Kolom sebelum perbaikan:")
print(df.isnull().sum())

# Mengisi missing values pada kolom 'Age' dengan mode (nilai paling sering muncul)
# df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].mean())   # Menggunakan mean perlu perbaikan
median_value = df['Age'].median()  # Menggunakan median
# mode_value = df['Age'].mode() # mode

# Mengisi missing values pada kolom 'Name' dengan "Unknown"
# fill value mean is not yet
df['Age'].fillna(median_value, inplace=True) #median
# df['Name'].fillna('Unknown', inplace=True) #mode

# Menampilkan data setelah perbaikan
print("\nData setelah penanganan missing values:")
print(df)

# Identifikasi duplikat
print("\nduplikat yang ditemukan: ",df.duplicated().sum())
# data duplikat yang akan dihapus
print("\n Data yang duplikat:")
print(df[df.duplicated()], '\n')
# Menghapus duplikat
df.drop_duplicates(inplace=True)
print(df)
