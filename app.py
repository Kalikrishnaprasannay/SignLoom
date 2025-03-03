import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import requests
from PIL import Image

# Load the trained model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model_transfer.keras")

model = load_model()

# Load class indices dynamically
@st.cache_resource
def load_class_indices():
    data_dir = "./SData"
    class_indices = {v: k for k, v in enumerate(sorted(os.listdir(data_dir)))}
    return {v: k for k, v in class_indices.items()}

index_to_class = load_class_indices()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to preprocess image
def preprocess_frame(frame):
    tensor = tf.image.resize(frame, [128, 128])
    tensor = tf.cast(tensor, tf.float32) / 255.0
    tensor = tf.expand_dims(tensor, axis=0)
    return tensor

# Function to send image for prediction via API
def get_prediction(image):
    try:
        # Convert image to model-compatible format
        processed_image = preprocess_frame(image)
        predictions = model.predict(processed_image)
        pred_index = np.argmax(predictions[0])
        pred_class = index_to_class.get(pred_index, "Unknown")
        confidence = np.max(predictions[0]) * 100
        return pred_class, confidence
    except Exception as e:
        return "Error", str(e)

# Streamlit UI
st.title("SignLoom")
st.sidebar.header("Options")
option = st.sidebar.radio("Choose an option:", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image of a hand sign", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert image for processing
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # API prediction
        pred_class, confidence = get_prediction(image)
        
        st.write(f"**Prediction:** {pred_class} ({confidence:.2f}%)")

elif option == "Use Webcam":
    st.write("Press 'Start' to access your webcam and detect hand signs.")
    run_webcam = st.button("Start Webcam")
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.write("Could not access webcam.")
                    break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)
                
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # API prediction
                    pred_class, confidence = get_prediction(rgb_frame)
                    
                    cv2.putText(frame, f'{pred_class} ({confidence:.2f}%)', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "No Hands Detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                stframe.image(frame, channels="BGR")
                
                if st.button("Stop Webcam"):
                    break
        
        cap.release()
