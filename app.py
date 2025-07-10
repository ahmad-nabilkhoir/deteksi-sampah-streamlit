import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import time

# Konstanta
IMG_SIZE = (128, 128)  # ← harus cocok dengan input model
CLASS_NAMES = ["organik 🍂", "anorganik 🗑", "daur ulang ♻️"]
CLASS_LABELS = ["organik", "anorganik", "daur_ulang"]
MODEL_PATH = Path("garbage_classifier.h5")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Konfigurasi Streamlit
st.set_page_config(page_title="♻️ Deteksi Jenis Sampah", layout="centered")
st.title("♻️ Deteksi Jenis Sampah Berbasis Gambar")

# Upload gambar
uploaded = st.file_uploader("📤 Unggah gambar sampah", type=["jpg", "jpeg", "png"])
camera = st.camera_input("📷 Atau ambil langsung dengan kamera")
img_data = uploaded or camera

if img_data:
    # Tampilkan gambar
    img = Image.open(img_data).convert("RGB")
    st.image(img, caption="📸 Gambar Input", use_column_width=True)

    # Preprocessing gambar
    img_arr = np.array(img)
    img_arr = cv2.resize(img_arr, IMG_SIZE)
    img_arr = img_arr / 255.0  # Normalisasi 0–1
    input_tensor = img_arr[None, ...]  # Tambah dimensi batch

    # Prediksi
    st.subheader("🔍 Memproses prediksi...")
    progress = st.progress(0)
    pred = model.predict(input_tensor)[0]
    idx = int(np.argmax(pred))
    confidence = float(pred[idx])

    # Progress bar animasi
    for i in range(int(confidence * 100)):
        time.sleep(0.01)
        progress.progress(i + 1)

    # Hasil prediksi
    st.success(f"🧠 Prediksi: **{CLASS_NAMES[idx]}**")
    st.write(f"📈 Tingkat keyakinan: `{confidence*100:.2f}%`")

    # Grafik probabilitas
    chart_data = {label: float(p) for label, p in zip(CLASS_LABELS, pred)}
    st.subheader("📊 Distribusi Probabilitas Kelas:")
    st.bar_chart(chart_data)
