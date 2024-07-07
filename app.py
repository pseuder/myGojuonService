from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow import keras
from PIL import Image, ImageOps
import skimage.transform
from io import BytesIO
from werkzeug.datastructures import FileStorage


from Hiragana.hiraganajapanese import label as hiragana_label

app = Flask(__name__)
CORS(app)

model = keras.models.load_model("Hiragana/hiragana.h5")

def preprocess_image(image_file):
    img = Image.open(image_file).convert("L") 
    img_resized = skimage.transform.resize(np.array(img), (48, 48))
    img_normalized = img_resized / np.max(img_resized)
    img_reshaped = img_normalized.reshape(1, 48, 48, 1)
    return img_reshaped


def predict_hiragana(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)
    predicted_hiragana = hiragana_label[predicted_index]
    return predicted_hiragana, float(prediction[0][predicted_index])


def invert_image(image):
    gray_image = ImageOps.grayscale(image)
    inverted_image = ImageOps.invert(gray_image)
    inverted_image.save("./uploads/inverted_image.png")
    return inverted_image


@app.route("/", methods=["GET"])
def index():
    return "Hello, world!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

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

            predicted_char, confidence = predict_hiragana(processed_image_file)

            print(predicted_char, confidence)
            return jsonify(
                {"predicted_hiragana": predicted_char, "confidence": confidence}
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
