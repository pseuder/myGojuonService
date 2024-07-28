from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow import keras
from PIL import Image, ImageOps
import skimage.transform
from io import BytesIO
from werkzeug.datastructures import FileStorage
import os

from Hiragana.hiraganajapanese import label as hiragana_label
from Katakana.katakanajapanese import label as katakana_label

app = Flask(__name__)
CORS(app)

# absolute path to the model file
abs_path = os.path.abspath(os.path.dirname(__file__))
model_path1 = os.path.join(abs_path, "Hiragana/hiragana.h5")
model_path2 = os.path.join(abs_path, "Katakana/katakana.h5")

hira_model = keras.models.load_model(model_path1)
kata_model = keras.models.load_model(model_path2)


def invert_image(image):
    gray_image = ImageOps.grayscale(image)
    inverted_image = ImageOps.invert(gray_image)
    inverted_image.save("./uploads/inverted_image.png")
    return inverted_image

def preprocess_image(image_file):
    img = Image.open(image_file).convert("L") 
    img_resized = skimage.transform.resize(np.array(img), (48, 48))
    img_normalized = img_resized / np.max(img_resized)
    img_reshaped = img_normalized.reshape(1, 48, 48, 1)
    return img_reshaped

def predict_charater(char_type, image):
    processed_image = preprocess_image(image)
    if char_type == "hiragana":
        model = hira_model
        label = hiragana_label
    elif char_type == "katakana":
        model = kata_model
        label = katakana_label
    else:
        return "Invalid character type", 0.0

    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)
    predicted_charater = label[predicted_index]
    return predicted_charater, float(prediction[0][predicted_index])


@app.route("/", methods=["GET"])
def index():
    return "Hello, world!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    char_type = request.form.get("char_type")

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if image_file:
        try:
            image = Image.open(image_file)
            inverted_image = invert_image(image)
            image_io = BytesIO()
            inverted_image.save(image_io, format=image.format)
            image_io.seek(0)

            processed_image_file = FileStorage(
                image_io,
                filename=image_file.filename,
                content_type=image_file.content_type,
            )

            predicted_char, confidence = predict_charater(char_type, processed_image_file)

            print(predicted_char, confidence)
            return jsonify(
                {"predicted_hiragana": predicted_char, "confidence": confidence}
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
