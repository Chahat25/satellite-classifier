print(">>> train_model.py is running!")
from preprocess import load_dataset
import tensorflow as tf
import os

# Parameters
DATASET_PATH = r"C:\Users\Chahat\OneDrive\Desktop\satellite-classifier\EuroSAT"  
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_SAVE_PATH = "saved_model/model.h5"

# Load dataset
print("[INFO] Loading dataset...")
train_ds, class_names = load_dataset(DATASET_PATH, IMG_SIZE, BATCH_SIZE)

# Build model
print("[INFO] Building model...")
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
print("[INFO] Starting training...")
model.fit(train_ds, epochs=EPOCHS)

# Save model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"[INFO] Model saved to {MODEL_SAVE_PATH}")
