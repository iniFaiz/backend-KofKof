# Backend KofKof - API Klasifikasi Batuk

Backend KofKof adalah aplikasi berbasis **FastAPI** yang menyediakan layanan klasifikasi suara batuk menggunakan model Machine Learning **TensorFlow**. API memproses file audio yang diunggah, mengubahnya menjadi spektrogram, dan memprediksi jenis batuknya.

## 🚀 Fitur Utama

* **API Cepat:** Dibangun menggunakan framework FastAPI untuk respons yang cepat.
* **Klasifikasi AI:** Menggunakan model `.h5` TensorFlow untuk analisis audio.
* **Pemrosesan Audio:** Menangani unggahan file audio, validasi header, dan konversi ke spektrogram secara otomatis.
* **Validasi Keamanan:** Membatasi ukuran file (maks 5MB) dan memverifikasi header file untuk keamanan.
* **Logging:** Mencatat riwayat prediksi termasuk nama file, hasil prediksi, tingkat kepercayaan (confidence), dan waktu pemrosesan.

## 🛠️ Teknologi yang Digunakan

Proyek ini menggunakan library Python berikut:

* **FastAPI & Uvicorn:** Server API modern dan asinkronus (async).
* **TensorFlow (CPU):** Untuk menjalankan model klasifikasi (inference).
* **Librosa & Soundfile:** Untuk pemrosesan dan analisis sinyal audio.
* **NumPy:** Untuk manipulasi data numerik.

## 📦 Instalasi

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/username-anda/backend-kofkof.git](https://github.com/username-anda/backend-kofkof.git)
    cd backend-kofkof
    ```

2.  **Siapkan Virtual Environment (Disarankan):**
    ```bash
    python -m venv venv
    # Aktifkan venv:
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows
    ```

3.  **Install Dependensi:**
    Pastikan Anda memiliki file `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Siapkan Model:**
    Pastikan file model (contoh: `cough_classifier_model.h5`) sudah diletakkan di direktori utama proyek sesuai konfigurasi.

## Menjalankan Aplikasi

Aplikasi akan berjalan di `http://0.0.0.0:8000`.
Atau gunakan `uvicorn` secara langsung (mode reload untuk development):

```bash
uvicorn main:app --reload
```

## 📡 Dokumentasi API

### 1. Cek Status (Health Check)
* **Endpoint:** `GET /`
* **Deskripsi:** Memastikan API berjalan dengan normal.

### 2. Prediksi Batuk (Predict)
* **Endpoint:** `POST /predict`
* **Deskripsi:** Mengunggah file audio untuk diklasifikasikan.
* **Body** (`multipart/form-data`):
  * `file`: File audio

**Proses:**
1. Validasi header file.
2. Simpan ke file sementara (*temp file*).
3. Pre-processing audio menjadi spektrogram.
4. Prediksi menggunakan model TensorFlow.
