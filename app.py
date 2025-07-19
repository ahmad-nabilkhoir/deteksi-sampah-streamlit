import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import time

# Load model
MODEL_PATH = Path("garbage_classifier.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# 4 kelas (organik, anorganik, daur ulang, bukan sampah)
CLASS_NAMES = ["organik 🍂", "anorganik 🗑", "daur ulang ♻️", "bukan sampah ❌"]
CLASS_LABELS = ["organik", "anorganik", "daur_ulang", "bukan_sampah"]
IMG_SIZE = (244, 244)  # Ukuran sesuai model training

# Konfigurasi Streamlit
st.set_page_config("♻️ Deteksi Jenis Sampah", layout="centered")
st.title("♻️ Deteksi Jenis Sampah Berbasis Gambar")

# Input gambar
uploaded = st.file_uploader("📄 Unggah gambar sampah", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="📸 Gambar Input", use_column_width=True)

    # Preprocessing
    img = img.resize(IMG_SIZE)
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)  # Bentuk (1, 244, 244, 3)

    # Prediksi
    pred = model.predict(img_arr)[0]  # Output probabilitas
    idx = int(np.argmax(pred))
    confidence = float(pred[idx])
    label = CLASS_NAMES[idx]

    # Progress bar
    st.subheader("🔍 Memproses prediksi...")
    progress = st.progress(0)
    for i in range(int(confidence * 100)):
        time.sleep(0.005)
        progress.progress(i + 1)

    # Logika deteksi
    if label == "bukan sampah ❌" or confidence < 0.70:
        st.error("❌ Gambar tidak dikenali sebagai jenis sampah.")
        st.write(f"📊 Tingkat keyakinan: `{confidence * 100:.2f}%`")
    else:
        st.success(f"🧠 Prediksi: **{label}**")
        st.write(f"📊 Tingkat keyakinan: `{confidence * 100:.2f}%`")

    # Tampilkan grafik probabilitas
    chart_data = {label: float(p) for label, p in zip(CLASS_LABELS, pred)}
    st.subheader("📊 Distribusi Probabilitas Kelas:")
    st.bar_chart(chart_data)
else:
    st.info("📅 Silakan unggah gambar terlebih dahulu.")
