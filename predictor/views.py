from django.shortcuts import render
from django.conf import settings

from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os


# =========================
# Load Model
# =========================

MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "model",
    "tomato_disease_efficientnetb0.keras"
)

CLASS_NAMES_PATH = os.path.join(
    settings.BASE_DIR,
    "model",
    "class_names.json"
)

model = load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


# =========================
# Home / Prediction
# =========================

def predict_disease(request):

    predicted_class = None
    confidence = None
    image_url = None

    if request.method == "POST":

        uploaded_image = request.FILES.get("image")

        if uploaded_image:

            # Open image
            img = Image.open(uploaded_image).convert("RGB")

            # Resize
            img = img.resize((224, 224))

            # Convert to NumPy
            img_array = np.array(img)

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            predictions = model.predict(img_array, verbose=0)

            # Get highest probability
            predicted_index = np.argmax(predictions[0])

            confidence = predictions[0][predicted_index] * 100

            predicted_class = class_names[predicted_index]

    return render(
        request,
        "predictor/index.html",
        {
            "predicted_class": predicted_class,
            "confidence": confidence,
        }
    )