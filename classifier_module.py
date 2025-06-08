# classifier_module.py

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# -------- Config --------
MODEL_PATH = os.path.join("model", "waste_model.h5")
LABEL_FILE = os.path.join("model", "labels.txt")  # Optional if it exists
IMAGE_SIZE = (224, 224)  # Must match your Teachable Machine export
NORMALIZE_DIVISOR = 255.0

# -------- Load Class Labels --------
def load_labels(label_file=LABEL_FILE):
    """Load class labels from labels.txt if available, else use defaults."""
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            labels = [line.strip() for line in f.readlines()]
            print(f"✅ Loaded {len(labels)} class labels from '{label_file}'")
            return labels
    else:
        print(f"⚠️ Label file '{label_file}' not found. Using default class labels.")
        return ['Plastic', 'Organic', 'Metal', 'Glass', 'Paper', 'E-waste']

CLASS_NAMES = load_labels()

# -------- Load Model --------
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# -------- Preprocess Image --------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert and normalize image for prediction."""
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    img_array = img_to_array(image) / NORMALIZE_DIVISOR  # Normalize to [0,1]
    return np.expand_dims(img_array, axis=0)

# -------- Predict Waste Type --------
def predict_image(image: Image.Image) -> str:
    """Predict the category of waste from the input image."""
    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image, verbose=0)[0]
        class_index = np.argmax(prediction)
        confidence = prediction[class_index] * 100
        label = CLASS_NAMES[class_index]
        return f"{label} ({confidence:.2f}% confidence)"
    except Exception as e:
        return f"Error during prediction: {e}"

