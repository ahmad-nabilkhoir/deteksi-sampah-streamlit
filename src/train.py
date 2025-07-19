import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
import numpy as np
from pathlib import Path

"""
TRAIN SCRIPT ― Deteksi Jenis Sampah (4 kelas)
=================================================
• Input size          : 244 × 244 px (lebih tajam)
• Batch size          : 64
• Epochs (top)        : 5 (freeze)  ➜  fine‑tune 5 lagi (unfreeze 50 lapisan)
• Augmentasi          : rotasi, zoom, shift, flip
• Class weights       : balanced (otomatis)
• Callback            : EarlyStopping + ReduceLROnPlateau
"""

# ────────────────────────────────────────────────
# Konfigurasi dasar
# ────────────────────────────────────────────────
IMG_SIZE        = (244, 244)
BATCH_SIZE      = 64
EPOCHS_BASE     = 5       # training head (base frozen)
EPOCHS_FINE     = 5       # fine‑tuning
DATA_DIR        = Path("data/train_val_split")
MODEL_SAVE_PATH = "garbage_classifier.h5"

# ────────────────────────────────────────────────
# Data generator + augmentasi
# ────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical",
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical",
)

num_classes = train_gen.num_classes
print("▶ Kelas terdeteksi:", train_gen.class_indices)

# Class weight untuk dataset tidak seimbang
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes,
)
class_weights = {i: w for i, w in enumerate(class_weights)}
print("▶ Class weights:", class_weights)

# ────────────────────────────────────────────────
# Model base + head
# ────────────────────────────────────────────────
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze dulu

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Callbacks
early_stop = EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
reduce_lr  = ReduceLROnPlateau(patience=2, factor=0.2, monitor="val_loss", verbose=1)

# ────────────────────────────────────────────────
# Train head
# ────────────────────────────────────────────────
model.fit(
    train_gen,
    epochs=EPOCHS_BASE,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
)

# ────────────────────────────────────────────────
# Fine‑tune – buka 50 lapisan terakhir
# ────────────────────────────────────────────────
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    train_gen,
    epochs=EPOCHS_FINE,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
)

# ────────────────────────────────────────────────
# Simpan model
# ────────────────────────────────────────────────
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Model selesai & tersimpan di {MODEL_SAVE_PATH}\n")
