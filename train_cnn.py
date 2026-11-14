from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).parent
TRAIN_DIR = ROOT / "data" / "fruit_dataset" / "train"
TEST_DIR = ROOT / "data" / "fruit_dataset" / "test"
OUT = ROOT / "models" / "freshness_cnn.keras"
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42

if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise SystemExit("Dataset folders missing. See README.md for the required directory structure.")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, validation_split=0.2,
    subset="training", label_mode="categorical"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, validation_split=0.2,
    subset="validation", label_mode="categorical"
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH, shuffle=False, label_mode="categorical"
)
class_names = train_ds.class_names
print("Class order:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

inputs = tf.keras.Input(shape=(224,224,3))
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = tf.keras.layers.RandomFlip("horizontal")(x)
x = tf.keras.layers.RandomRotation(0.08)(x)
for filters in [32,64,128]:
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(192, 3, padding="same", activation="relu", name="last_conv")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.35)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss"),
    tf.keras.callbacks.ModelCheckpoint(OUT, save_best_only=True, monitor="val_loss")
]
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)
loss, acc = model.evaluate(test_ds, verbose=0)
print(f"Test loss: {loss:.4f} | Test accuracy: {acc:.4f}")
model.save(OUT)
print(f"Saved model: {OUT}")
