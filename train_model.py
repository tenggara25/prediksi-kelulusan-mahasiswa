# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Contoh dataset
data = pd.DataFrame({
    'ipk': [3.4, 2.5, 3.1, 2.2, 3.7, 2.8, 3.0, 3.2],
    'sks_lulus': [90, 72, 88, 60, 96, 70, 80, 85],
    'presensi': [92, 70, 85, 65, 94, 75, 80, 88],
    'mengulang': [0, 2, 1, 3, 0, 2, 1, 0],
    'lulus_tepat_waktu': [1, 0, 1, 0, 1, 0, 1, 1]
})

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
