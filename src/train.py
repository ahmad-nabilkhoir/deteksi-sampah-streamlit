import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

IMG_SIZE = (224, 224)
BATCH = 32
ROOT = Path("data/train_val_split")

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    horizontal_flip=True,
    validation_split=0.2)

train_ds = train_gen.flow_from_directory(
    ROOT,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    subset="training")

val_ds = train_gen.flow_from_directory(
    ROOT,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    subset="validation")

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_ds, epochs=10, validation_data=val_ds)

# Fine-tuning
base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_ds, epochs=5, validation_data=val_ds)

model.save("garbage_classifier.h5")
