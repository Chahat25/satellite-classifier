import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Add class labels in the correct order for EuroSAT (or your dataset)
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

def predict_single_image(model_path, image_path, image_size=64):
    model = load_model(model_path)

    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (image_size, image_size))
    img_input = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension

    prediction = model.predict(img_input)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]

    print(f"Predicted class index: {class_index}")
    print(f"Predicted class label: {class_label}")

    # Optional: Display image with label
    cv2.putText(img, f"Prediction: {class_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Predicted Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = r"C:/Users/Chahat/OneDrive/Desktop/satellite-classifier/saved_model/model.h5"
    image_path = r"C:\Users\Chahat\OneDrive\Desktop\satellite-classifier\EuroSAT\Forest\Forest_4.jpg"  # update if needed

    predict_single_image(model_path, image_path)
