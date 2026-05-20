from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Load model
model = load_model("brain_tumor_model.h5")
print("Model loaded")

# Class names must match your training folder names
CLASS_NAMES  = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE   = 128
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(img_path):
    img       = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions         = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence          = round(float(np.max(predictions)) * 100, 2)
    class_name          = CLASS_NAMES[predicted_class_idx]

    if class_name == "notumor":
        result = "No Tumor"
    else:
        result = "Tumor Detected: " + class_name.capitalize()

    return result, confidence

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"})

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    result, confidence = predict_image(filepath)

    return jsonify({
        "prediction" : result,
        "confidence" : str(confidence) + "%"
    })

if __name__ == "__main__":
    app.run(debug=True)
