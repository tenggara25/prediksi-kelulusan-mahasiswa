# app.py
from flask import Flask, render_template, request
from joblib import load
import pandas as pd

app = Flask(__name__)

# ==========================
# 1. Load model yang sudah dilatih
# ==========================
# Pastikan file model_kelulusan.joblib ada di folder yang sama
model = load("model_kelulusan.joblib")


# ==========================
# 2. Fungsi logika prediksi
# ==========================
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
    label = "Lulus tepat waktu" if y_pred == 1 else "Berisiko terlambat"

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


# ==========================
# 3. Route web utama
# ==========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None  # default: belum ada hasil
    error = None

    if request.method == "POST":
        try:
            # Ambil data dari form (tambahkan nama & nim)
            nama = request.form.get("nama", "").strip()
            nim = request.form.get("nim", "").strip()

            # Ambil fitur numerik
            ipk = float(request.form["ipk"])
            sks_lulus = int(request.form["sks_lulus"])
            presensi = int(request.form["presensi"])
            mengulang = int(request.form["mengulang"])

            # (opsional) validasi sederhana
            if not nama:
                raise ValueError("Nama tidak boleh kosong.")
            if not nim:
                raise ValueError("NIM tidak boleh kosong.")
            if not (0.0 <= ipk <= 4.0):
                raise ValueError("IPK harus antara 0.0 - 4.0.")
            if not (0 <= presensi <= 100):
                raise ValueError("Presensi harus antara 0 - 100.")
            if sks_lulus < 0:
                raise ValueError("SKS Lulus harus >= 0.")
            if mengulang < 0:
                raise ValueError("Jumlah mengulang harus >= 0.")

            # Panggil fungsi prediksi
            label, prob_percent, rekomendasi = prediksi_kelulusan(
                ipk, sks_lulus, presensi, mengulang
            )

            # Data yang dikirim ke template (index.html)
            result = {
                "nama": nama,
                "nim": nim,
                "label": label,
                "probabilitas": f"{prob_percent:.2f}%",
                "prob_value": max(0, min(prob_percent, 100)),  # untuk lebar progress bar
                "rekomendasi": rekomendasi,
                # ringkasan input user
                "ipk": ipk,
                "sks_lulus": sks_lulus,
                "presensi": presensi,
                "mengulang": mengulang,
            }

        except Exception as e:
            # Tangani error input / prediksi agar tidak crash
            error = str(e)
            result = None

    return render_template("index.html", result=result, error=error)


# ==========================
# 4. Run app
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
