# Contoh Data Cleaning dengan Python
import pandas as pd
import numpy as np

# Membuat dataframe contoh
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Eve', np.nan],
    'Age': [25, 30, 35, 25, np.nan, 50],
    'City': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Miami', 'Los Angeles']
}
df = pd.DataFrame(data)

# Menampilkan data awal
print("Data Awal:")
print(df)

# Menghapus duplikat
df = df.drop_duplicates()

# Menangani missing values dengan mengisi nilai median untuk kolom numerik
df['Age'] = df['Age'].fillna(df['Age'].median())

# Menghapus baris yang mengandung missing values di kolom 'Name'
df = df.dropna(subset=['Name'])

# Menampilkan data setelah cleaning
print("\nData Setelah Cleaning:\n",df)
