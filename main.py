#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree

#baca file
dy = pd.read_csv("pilkada-100-uas.csv")
df = pd.read_csv("pilkada-1000-uas.csv")

#melihat file
df

#ubah data non-numerik menjadi numerik
le = LabelEncoder()

df['Nama Daerah'] = le.fit_transform(df['Nama Daerah'])
df['Jalur Pencalonan'] = le.fit_transform(df['Jalur Pencalonan'])
df['Nama Kandidat (1)'] = le.fit_transform(df['Nama Kandidat (1)'])
df['Nama Kandidat (2)'] = le.fit_transform(df['Nama Kandidat (2)'])
df['Gender (1)'] = le.fit_transform(df['Gender (1)'])
df['Gender (2)'] = le.fit_transform(df['Gender (2)'])
df['Latar Belakang Profesi (1)'] = le.fit_transform(df['Latar Belakang Profesi (1)'])
df['Latar Belakang Profesi (2)'] = le.fit_transform(df['Latar Belakang Profesi (2)'])
df['Hasil Pilkada'] = le.fit_transform(df['Hasil Pilkada'])

dy['Nama Daerah'] = le.fit_transform(dy['Nama Daerah'])
dy['Jalur Pencalonan'] = le.fit_transform(dy['Jalur Pencalonan'])
dy['Nama Kandidat (1)'] = le.fit_transform(dy['Nama Kandidat (1)'])
dy['Nama Kandidat (2)'] = le.fit_transform(dy['Nama Kandidat (2)'])
dy['Gender (1)'] = le.fit_transform(dy['Gender (1)'])
dy['Gender (2)'] = le.fit_transform(dy['Gender (2)'])
dy['Latar Belakang Profesi (1)'] = le.fit_transform(dy['Latar Belakang Profesi (1)'])
dy['Latar Belakang Profesi (2)'] = le.fit_transform(dy['Latar Belakang Profesi (2)'])
dy['Hasil Pilkada'] = le.fit_transform(dy['Hasil Pilkada'])

#lihat data
df

#lihat  korelasi untuk memutuskan data yang tidak berguna
for kolom in df.columns:
    if kolom != 'Hasil Pilkada':
        korelasi = df['Hasil Pilkada'].corr(df[kolom])
        print(f"korelasi Kolom 'Hasil Pilkada' dengan kolom '{kolom}': {korelasi}")

#lewati data yang korelasinya jelek berdasarkan penelusuran data pilkada 1000
df = df.drop(['Jumlah Paslon'], axis=1)
df = df.drop(['Jalur Pencalonan'], axis=1)
df = df.drop(['Nama Kandidat (1)', 'Nama Kandidat (2)'], axis=1)
df = df.drop(['Gender (1)', 'Gender (2)'], axis=1)
df = df.drop(['Usia (1)', 'Usia (2)'], axis=1)
df = df.drop(['Latar Belakang Profesi (1)', 'Latar Belakang Profesi (2)'], axis=1)

dy = dy.drop(['Jumlah Paslon'], axis=1)
dy = dy.drop(['Jalur Pencalonan'], axis=1)
dy = dy.drop(['Nama Kandidat (1)', 'Nama Kandidat (2)'], axis=1)
dy = dy.drop(['Gender (1)', 'Gender (2)'], axis=1)
dy = dy.drop(['Usia (1)', 'Usia (2)'], axis=1)
dy = dy.drop(['Latar Belakang Profesi (1)', 'Latar Belakang Profesi (2)'], axis=1)

#lihat data
df

#memastikan data sebelum diproses
#menangani str pada file 1000 data
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = LabelEncoder().fit_transform(df[col])

#menangani str pada file 100 data
for col in dy.columns:
    if dy[col].dtype == "object":
        dy[col] = LabelEncoder().fit_transform(dy[col])

#melihat tipe data sebelum training (memastikan)
df.dtypes

#training
X_train_1000, X_test_1000, y_train_1000, y_test_1000 = train_test_split(df.drop("Hasil Pilkada", axis=1), df["Hasil Pilkada"], test_size=0.2)

model = DecisionTreeClassifier(max_depth=3, criterion="entropy")
model.fit(X_train_1000, y_train_1000)

#cek akurasi training
prediksi = model.predict(X_test_1000)
akurasi = np.mean(prediksi == y_test_1000)
print("peforma training 1000 data:", akurasi)

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=df.drop("Hasil Pilkada", axis=1).columns, filled=True)
plt.show()

X_train_100, X_test_100, y_train_100, y_test_100 = train_test_split(dy.drop("Hasil Pilkada", axis=1), dy["Hasil Pilkada"], test_size=0.2)
y_pred_100 = model.predict(X_test_100)

#hitung akurasi analisi
akurasiimplementasi = np.mean(y_pred_100 == y_test_100)
print("Akurasi hasil:", akurasiimplementasi)