# app.py
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import keras
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('mnist.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    image = Image.open(file.stream).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image)  # Convert to numpy array
    image = image.reshape(1, 28, 28, 1)  # Reshape for the model
    image = image / 255.0  # Normalize

    # Predict the digit
    prediction = model.predict(image)[0]
    digit = np.argmax(prediction)
    confidence = prediction.tolist()  # Confidence distribution for all digits

    # Return the result as JSON
    return jsonify({'digit': int(digit), 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)