import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from preprocess import load_dataset

model = tf.keras.models.load_model("saved_model/eurosat_cnn.h5")
dataset_path = "EuroSAT"
img_size = (64, 64)
batch_size = 32

test_ds, class_names = load_dataset(dataset_path, img_size, batch_size)

def plot_predictions(model, dataset, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        predictions = model.predict(images)
        predicted_labels = tf.argmax(predictions, axis=1)
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[predicted_labels[i]]}")
            plt.axis("off")
    plt.tight_layout()
    plt.show()

plot_predictions(model, test_ds, class_names)
