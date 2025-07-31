# 🛰️ Satellite Image Classifier using CNN (EuroSAT Dataset)

This project implements a Convolutional Neural Network (CNN) to classify satellite images based on land use and land cover using the [EuroSAT dataset](https://github.com/phelber/eurosat). The model is built using TensorFlow/Keras and can classify remote sensing images into multiple categories like forest, residential, industrial, etc.

---

## 📁 Project Structure
satellite-classifier/
├── EuroSAT/ # (Add full dataset manually – not uploaded to GitHub)
├── saved_model/ # Trained model (.h5)
├── dataset/ # (Optional: a small sample for demo/testing)
├── src/ # Source code
│ ├── init.py
│ ├── preprocess.py # Image preprocessing and dataset split
│ ├── train_model.py # CNN training script
│ ├── evaluate.py # Model evaluation and accuracy report
│ └── predict_single.py # Predict single image using trained model
├── eurosat_cnn.ipynb # Jupyter notebook version (optional)
├── requirements.txt # Python dependencies
├── .gitignore # Files to ignore in Git
└── README.md # Project overview and usage instructions

yaml
Copy
Edit

---

## 🔧 Setup Instructions

### 1. Install Dependencies

Make sure you have Python ≥ 3.8 and run:

```bash
pip install -r requirements.txt
2. Add Dataset
Download the EuroSAT RGB dataset:
EuroSAT RGB.zip (Direct Download)

Extract it into the EuroSAT/ directory (this folder is excluded from Git).

🚀 Usage
🧹 Preprocess the Dataset
bash
Copy
Edit
python src/preprocess.py
🏋️‍♀️ Train the Model
bash
Copy
Edit
python src/train_model.py
This will save the model to:
saved_model/satellite_cnn_model.h5

📊 Evaluate the Model
bash
Copy
Edit
python src/evaluate.py
🔍 Predict a Single Image
bash
Copy
Edit
python src/predict_single.py
✏️ Make sure to update the image_path in predict_single.py to test a local image.

🧠 Model Details
Type: CNN (Convolutional Neural Network)

Input: RGB images (64x64)

Framework: TensorFlow & Keras

Classes: 10 EuroSAT land-use categories

✅ Sample Output
kotlin
Copy
Edit
Predicted class index: 2
(You can map class indices using the official EuroSAT class labels.)

💡 Future Improvements
Use .keras format for model saving (Keras 3 standard)

Add confusion matrix and class-wise accuracy

Build a web-based prediction interface

👩‍💻 Author
Chahat Gupta
July 2025
AI + Remote Sensing | EuroSAT Classifier

📝 License
This project is for educational and academic use only.
Dataset © by EuroSAT under the open license.
