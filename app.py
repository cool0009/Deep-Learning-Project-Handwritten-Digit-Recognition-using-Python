from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from keras.models import load_model
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mnist.h5"

app = Flask(__name__)
model = load_model(MODEL_PATH)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "main.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def preprocess_image(file_storage):
    image = Image.open(file_storage.stream)
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)

    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image = np.array(image, dtype=np.float32) / 255.0
    return image.reshape(1, 28, 28, 1)


def preprocess_pixels(pixels):
    """Process pixel array from canvas (already normalized 0-1)."""
    # Reshape to 28x28 if needed
    if len(pixels) == 784:
        image = np.array(pixels).reshape(28, 28)
    else:
        # Assume it's already 28x28
        image = np.array(pixels).reshape(28, 28)
    return image.reshape(1, 28, 28, 1)


@app.route("/predict", methods=["POST"])
def predict():
    # Check for JSON format (new canvas)
    if request.is_json:
        data = request.get_json()
        if "pixels" in data:
            image = preprocess_pixels(data["pixels"])
            prediction = model.predict(image, verbose=0)[0]
            digit = int(np.argmax(prediction))
            return jsonify({"digit": digit, "confidence": prediction.tolist()})
    
    # Check for file upload (old format)
    if "image" in request.files:
        image = preprocess_image(request.files["image"])
        prediction = model.predict(image, verbose=0)[0]
        digit = int(np.argmax(prediction))
        return jsonify({"digit": digit, "confidence": prediction.tolist()})

    return jsonify({"error": "Missing image file or pixels"}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
