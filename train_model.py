# train_model.py
"""
Script sederhana untuk melatih model prediksi kelulusan mahasiswa
dan menyimpannya ke file model_kelulusan.joblib

Jalankan:
    python train_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# ==========================
# 1. Siapkan dataset
# ==========================
# Contoh dataset dummy (silakan ganti dengan dataset asli kampus jika ada)

data = pd.DataFrame({
    "ipk":        [3.4, 2.5, 3.1, 2.2, 3.7, 2.8, 3.0, 2.4, 3.3, 2.6],
    "sks_lulus":  [90,   72,  88,  60,  96,  70,  80,  65,  92,  74],
    "presensi":   [92,   70,  85,  65,  94,  75,  80,  68,  90,  72],
    "mengulang":  [0,    2,   1,   3,   0,   2,   1,   3,   0,   2],
    # 1 = lulus tepat waktu, 0 = terlambat / berisiko
    "lulus_tepat_waktu": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
})

# Jika nanti pakai file CSV, contoh:
# data = pd.read_csv("data_mahasiswa.csv")

X = data[["ipk", "sks_lulus", "presensi", "mengulang"]]
y = data["lulus_tepat_waktu"]

# ==========================
# 2. Split train–test (opsional, untuk evaluasi)
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ==========================
# 3. Training model
# ==========================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# ==========================
# 4. Evaluasi sederhana
# ==========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Akurasi pada data test: {acc:.2f}")

# ==========================
# 5. Simpan model
# ==========================
dump(model, "model_kelulusan.joblib")
print("Model berhasil disimpan ke: model_kelulusan.joblib")
