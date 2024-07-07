import numpy as np
from tensorflow import keras
from PIL import Image
import skimage.transform

# Load the trained model
model = keras.models.load_model("hiragana.h5")

# Load the labels
label = [
    "あ",
    "い",
    "う",
    "え",
    "お",
    "か",
    "が",
    "き",
    "ぎ",
    "く",
    "ぐ",
    "け",
    "げ",
    "こ",
    "ご",
    "さ",
    "ざ",
    "し",
    "じ",
    "す",
    "ず",
    "せ",
    "ぜ",
    "そ",
    "ぞ",
    "た",
    "だ",
    "ち",
    "ぢ",
    "つ",
    "づ",
    "て",
    "で",
    "と",
    "ど",
    "な",
    "に",
    "ぬ",
    "ね",
    "の",
    "は",
    "ば",
    "ぱ",
    "ひ",
    "び",
    "ぴ",
    "ふ",
    "ぶ",
    "ぷ",
    "へ",
    "べ",
    "ぺ",
    "ほ",
    "ぼ",
    "ぽ",
    "ま",
    "み",
    "む",
    "め",
    "も",
    "や",
    "ゆ",
    "よ",
    "ら",
    "り",
    "る",
    "れ",
    "ろ",
    "わ",
    "を",
    "ん",
]


def preprocess_image(image_path):
    # Open the image
    img = Image.open(image_path).convert("L")  # Convert to grayscale

    # Resize the image to 48x48
    img_resized = skimage.transform.resize(np.array(img), (48, 48))

    # Normalize the image
    img_normalized = img_resized / np.max(img_resized)

    # Reshape for model input
    img_reshaped = img_normalized.reshape(1, 48, 48, 1)

    return img_reshaped


def predict_hiragana(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(processed_image)

    # Get the index of the highest probability
    predicted_index = np.argmax(prediction)

    # Get the corresponding hiragana character
    predicted_hiragana = label[predicted_index]

    return predicted_hiragana, prediction[0][predicted_index]


for i in range(1, 6):
    image_path = f"test_data/{i}.png"  # Replace with your image path
    predicted_char, confidence = predict_hiragana(image_path)
    print(f"Predicted Hiragana: {predicted_char}")
    print(f"Confidence: {confidence:.2f}")
