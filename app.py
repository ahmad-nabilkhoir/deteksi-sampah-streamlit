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
CLASS_NAMES = ["organik 🍂", "anorganik 🗑", "daur ulang ♻️"]
CLASS_LABELS = ["organik", "anorganik", "daur_ulang"]

# Konfigurasi Streamlit
st.set_page_config("♻️ Deteksi Jenis Sampah", layout="centered")
st.title("♻️ Deteksi Jenis Sampah Berbasis Gambar")

# Input gambar
uploaded = st.file_uploader("📤 Unggah gambar sampah", type=["jpg", "jpeg", "png"])
camera = st.camera_input("📷 Atau ambil langsung dengan kamera")

img_data = uploaded or camera

if img_data:
    img = Image.open(img_data).convert("RGB")
    st.image(img, caption="📸 Gambar Input", use_column_width=True)

    # Preprocessing
    img_arr = np.array(img)
    img_arr = cv2.resize(img_arr, (224, 224))
    img_arr = img_arr / 255.0
    pred = model.predict(img_arr[None, ...])[0]  # shape: (3,)
    idx = int(np.argmax(pred))
    confidence = float(pred[idx])

    # Progress bar
    st.subheader("🔍 Memproses prediksi...")
    progress = st.progress(0)
    for i in range(int(confidence * 100)):
        time.sleep(0.01)
        progress.progress(i + 1)

    # Hasil
    st.success(f"🧠 Prediksi: **{CLASS_NAMES[idx]}**")
    st.write(f"📈 Tingkat keyakinan: `{confidence*100:.2f}%`")

    # Bar Chart
    chart_data = {label: float(p) for label, p in zip(CLASS_LABELS, pred)}
    st.subheader("📊 Distribusi Probabilitas Kelas:")
    st.bar_chart(chart_data)
