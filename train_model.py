# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Contoh dataset
data = pd.read_csv("dataset_kelulusan_mahasiswa.csv")

X = data[['ipk','sks_lulus','presensi','mengulang']]
y = data['lulus_tepat_waktu']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Simpan model
dump(model, "model_kelulusan.joblib")
print("Model tersimpan sebagai model_kelulusan.joblib")
