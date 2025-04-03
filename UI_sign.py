import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import tempfile
import av
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ---- Load the trained model with error handling ----
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("sign_language_model_transfer.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ---- Load class indices ----
import os
class_indices = {v: k for k, v in enumerate(sorted(os.listdir("./SData")))}
index_to_class = {v: k for k, v in class_indices.items()}

# ---- Initialize MediaPipe Hand Detection ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---- Function to generate speech using gTTS ----
def speak_text(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            st.audio(tmp_file.name, format="audio/mp3")
    except Exception as e:
        st.error(f"Speech synthesis error: {e}")

# ---- Faster Frame Preprocessing ----
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = cv2.resize(frame, (128, 128))  # Resize to match model input
    frame = np.array(frame, dtype=np.float32) / 255.0  # Normalize
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# ---- Predict Sign Function ----
def predict_sign(frame):
    if model is None:
        return "Error: Model not loaded", 0.0

    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)
    pred_index = np.argmax(predictions[0])
    pred_class = index_to_class.get(pred_index, "Unknown")
    confidence = np.max(predictions[0]) * 100
    return pred_class, confidence

# ---- WebRTC Video Processing Class ----
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ---- Predict sign language only if a hand is detected ----
            if model:
                pred_class, confidence = predict_sign(img_rgb)

                # ---- Display Prediction on Stream ----
                cv2.putText(img, f'{pred_class} ({confidence:.2f}%)', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # ---- Speak Detected Sign ----
                speak_text(pred_class.replace("_", " "))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---- STREAMLIT UI ----
st.title("SignLoom: Real-Time Sign Language Interpreter")

# ---- Webcam or Video Upload Option ----
option = st.radio("Choose input method:", ("Use Webcam", "Upload Video"))

if option == "Use Webcam":
    st.write("Real-time Sign Language Recognition")
    webrtc_streamer(key="sign-detection", video_processor_factory=VideoProcessor)

elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        st.video("temp_video.mp4")

        cap = cv2.VideoCapture("temp_video.mp4")
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Predict Sign
            pred_class, confidence = predict_sign(frame)

            # Speak the predicted sign
            speak_text(pred_class.replace("_", " "))

            # Display Prediction on Frame
            cv2.putText(frame, f'{pred_class} ({confidence:.2f}%)', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            stframe.image(frame, channels="BGR")

        cap.release()
