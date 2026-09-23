# 🍅 NABTA

## AI-Based Tomato Leaf Disease Detection

NABTA is a Deep Learning web application that detects tomato leaf diseases from images using Computer Vision and EfficientNetB0.

### 📌 Project Overview

The project uses the PlantVillage dataset to train a tomato leaf disease classification model.

The trained EfficientNetB0 model was integrated into a Django web application, allowing users to upload a tomato leaf image and receive a predicted disease with its confidence score.

### 🧠 Model

- Model: EfficientNetB0
- Transfer Learning
- Fine-Tuning
- Input Size: 224 × 224
- Number of Classes: 10
- Validation Accuracy: ~96%

### 🌱 Supported Classes

- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Healthy

### 🛠️ Technologies

- Python
- TensorFlow / Keras
- EfficientNetB0
- NumPy
- Pandas
- Scikit-learn
- Pillow
- Django
- HTML
- CSS

### 🖥️ Application

The application allows the user to:

1. Upload a tomato leaf image.
2. Process the image using the trained model.
3. Predict the disease.
4. Display the predicted class and confidence score.

### 📊 Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The model achieved approximately **96% validation accuracy** after fine-tuning.

> Note: This is validation accuracy, not accuracy on a completely independent test set.

### 🚀 Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

Then open:

http://127.0.0.1:8000/
👩‍💻 Author

Alaa Mohamed

AI & Data Science Student
Machine Learning | Deep Learning | Computer Vision
