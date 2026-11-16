# Dashboard SER IndoWaveSentiment

## Alur dari notebook ke dashboard
1. Tiap anggota menjalankan notebooknya (Anggota 1 MFCC, 2 Log-Mel, 3 Prosodik), lalu menyimpan folder `results/`, `models/`, `src/` sebagai **Dataset Kaggle** (Save Version > Output, atau Create Dataset).
2. Buka `SER_Evaluasi_Gabungan.ipynb`, *Add Input* ketiga dataset tersebut + dataset IndoWaveSentiment, jalankan semua sel.
   Hasilnya: tabel Lapis 1/2/3, ablasi, model terpilih, dan folder/zip `dashboard_package/` (model terpilih + `assets/` + `requirements.txt`).
3. Salin `app.py` dan `app_utils.py` ke dalam `dashboard_package/`.
4. Uji lokal: `pip install -r requirements.txt` lalu `streamlit run app.py`.
5. Deploy: push isi `dashboard_package/` ke GitHub, lalu Streamlit Community Cloud atau Hugging Face Spaces (SDK Streamlit).

## Struktur
```
dashboard_package/
  app.py, app_utils.py, requirements.txt
  model/    model_final.joblib  (atau model_final.keras + preprocessor.joblib)
  src/      preprocess_<metode>.py, infer.py      # fungsi fitur yang SAMA dengan saat pelatihan
  assets/   results_all.csv, ablation_all.csv, leaderboard.csv, oof_all.csv, dataset_meta.csv, selection.json, samples/
```

## Isi dashboard
- **Prediksi:** unggah/rekam suara -> emosi, keyakinan, sentimen, plus pitch, energi, dan spektrogram rekamanmu.
- **Perbandingan model:** heatmap metode x model, leaderboard, ablasi.
- **Analisis kesalahan:** confusion matrix interaktif (5 emosi / 3 sentimen), F1 per kelas, akurasi per gender/intensitas/aktor.
- **Tantangan emosi:** coba "mengelabui" model dengan emosi target dan kumpulkan skor.
- **Tentang dataset:** ringkasan, contoh audio, sitasi, lisensi, keterbatasan.

## Catatan
- Jika model terpilih berjenis Keras, `requirements.txt` memuat `tensorflow-cpu` (berat; Hugging Face Spaces lebih aman).
- Lisensi dataset CC BY-NC 4.0: boleh untuk tugas non-komersial dengan atribusi. Cek lagi sebelum menyertakan contoh audio di repo publik.
