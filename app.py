import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import pyttsx3
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("sign_language_model_transfer.keras")

# Load class indices
class_indices = {v: k for k, v in enumerate(sorted(os.listdir("./SData")))}
index_to_class = {v: k for k, v in class_indices.items()}

# Initialize text-to-speech engine
engine = pyttsx3.init()

def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_sign(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    pred_index = np.argmax(predictions[0])
    pred_class = index_to_class.get(pred_index, "Unknown")
    confidence = np.max(predictions[0]) * 100
    return pred_class, confidence

def main():
    st.title("Sign Language Recognition")
    st.sidebar.title("Options")
    mode = st.sidebar.radio("Choose Input Mode:", ["Upload Image", "Use Webcam"])
    speak_enabled = st.sidebar.checkbox("Enable Text-to-Speech", True)
    
    if mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            pred_class, confidence = predict_sign(image)
            st.success(f"Predicted Sign: {pred_class} ({confidence:.2f}%)")
            if speak_enabled:
                engine.say(pred_class)
                engine.runAndWait()
    
    elif mode == "Use Webcam":
        st.write("Webcam integration is under development.")

if __name__ == "__main__":
    main()
