# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# =========================
# 1. DATASET CONTOH & MODEL
# =========================
# Di real project, bagian ini sebaiknya diganti dengan dataset asli kampus
# atau model yang sudah dilatih terpisah dan disimpan (joblib/pkl).

# Dataset dummy sederhana
data = pd.DataFrame({
    "ipk":        [3.4, 2.5, 3.1, 2.2, 3.7, 2.8, 3.0, 2.4],
    "sks_lulus":  [90,   72,  88,  60,  96,  70,  80,  65],
    "presensi":   [92,   70,  85,  65,  94,  75,  80,  68],
    "mengulang":  [0,    2,   1,   3,   0,   2,   1,   3],
    # 1 = lulus tepat waktu, 0 = terlambat / berisiko
    "lulus_tepat_waktu": [1, 0, 1, 0, 1, 0, 1, 0]
})

X = data[["ipk", "sks_lulus", "presensi", "mengulang"]]
y = data["lulus_tepat_waktu"]

# Training model RandomForest sederhana
model = RandomForestClassifier(random_state=42)
model.fit(X, y)


# =========================
# 2. FUNGSI PREDIKSI
# =========================
def prediksi_kelulusan(ipk, sks_lulus, presensi, mengulang):
    """
    Mengembalikan:
      - label (string): "Lulus tepat waktu" / "Berisiko terlambat"
      - prob_percent (float): probabilitas (%) lulus tepat waktu
      - rekomendasi (string): teks rekomendasi tindakan
    """
    df_input = pd.DataFrame([{
        "ipk": ipk,
        "sks_lulus": sks_lulus,
        "presensi": presensi,
        "mengulang": mengulang,
    }])

    # Prediksi kelas 0 / 1
    y_pred = model.predict(df_input)[0]

    # Probabilitas kelas "lulus tepat waktu" (diasumsikan label 1)
    prob_lulus = model.predict_proba(df_input)[0][1]  # nilai 0–1
    prob_percent = prob_lulus * 100

    # Label teks
    if y_pred == 1:
        label = "Lulus tepat waktu"
    else:
        label = "Berisiko terlambat"

    # Rekomendasi berdasarkan probabilitas
    if prob_percent >= 85:
        rekomendasi = (
            "Pertahankan performa. Tetap jaga IPK, presensi, dan konsistensi belajar."
        )
    elif prob_percent >= 60:
        rekomendasi = (
            "Perlu sedikit peningkatan. Tingkatkan presensi, atur jadwal belajar, "
            "dan konsultasi dengan dosen PA bila perlu."
        )
    else:
        rekomendasi = (
            "Wajib ikut mentoring / bimbingan intensif. Fokus perbaiki IPK, "
            "kurangi mengulang mata kuliah, dan tingkatkan kehadiran."
        )

    return label, prob_percent, rekomendasi


# =========================
# 3. ROUTE WEB UTAMA
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None  # default: belum ada hasil

    if request.method == "POST":
        # Ambil data dari form
        ipk = float(request.form["ipk"])
        sks_lulus = int(request.form["sks_lulus"])
        presensi = int(request.form["presensi"])
        mengulang = int(request.form["mengulang"])

        # Panggil fungsi prediksi
        label, prob_percent, rekomendasi = prediksi_kelulusan(
            ipk, sks_lulus, presensi, mengulang
        )

        # Data yang dikirim ke template
        result = {
            "label": label,
            "probabilitas": f"{prob_percent:.2f}%",
            "prob_value": max(0, min(prob_percent, 100)),  # untuk lebar progress bar (0–100)
            "rekomendasi": rekomendasi,
            # ringkasan input user
            "ipk": ipk,
            "sks_lulus": sks_lulus,
            "presensi": presensi,
            "mengulang": mengulang,
        }

    return render_template("index.html", result=result)


# =========================
# 4. RUN APP
# =========================
if __name__ == "__main__":
    # Untuk production, ganti debug=False
    app.run(debug=True)
