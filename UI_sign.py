import streamlit as st
import numpy as np
import os
import cv2
import tempfile
import av
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Ensure OpenCV uses headless mode to avoid libGL issues
cv2.setNumThreads(1)

# Load the trained model with error handling
@st.cache_resource  # Cache the model for better performance
def load_model():
    try:
        return tf.keras.models.load_model("sign_language_model_transfer.keras")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Load class indices from dataset
class_indices = {v: k for k, v in enumerate(sorted(os.listdir("./SData")))}
index_to_class = {v: k for k, v in class_indices.items()}

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to generate speech from text using gTTS
def speak_text(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            st.audio(tmp_file.name, format="audio/mp3")
    except Exception as e:
        st.error(f"Speech synthesis error: {e}")

# Preprocess Frame Function
def preprocess_frame(frame):
    tensor = cv2.resize(frame, (128, 128))
    tensor = np.array(tensor, dtype=np.float32) / 255.0  # Normalize
    tensor = np.expand_dims(tensor, axis=0)  # Add batch dimension
    return tensor

# Function to Predict Sign
def predict_sign(frame):
    if model is None:
        return "Error: Model not loaded", 0.0
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)
    pred_index = np.argmax(predictions[0])
    pred_class = index_to_class.get(pred_index, "Unknown")
    confidence = np.max(predictions[0]) * 100
    return pred_class, confidence

# WebRTC Video Processing Class
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

            # Predict sign if a hand is detected
            if model:
                pred_class, confidence = predict_sign(img_rgb)
                cv2.putText(img, f'{pred_class} ({confidence:.2f}%)', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Speak detected sign
                speak_text(pred_class.replace("_", " "))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---- STREAMLIT UI ----
st.title("SignLoom: Sign Language Interpreter")

# Option to use Webcam or Upload Video
option = st.radio("Choose input method:", ("Use Webcam", "Upload Video"))

# Placeholder for detected sign
detected_sign_placeholder = st.empty()

if option == "Use Webcam":
    st.write("Real-time Sign Language Recognition")
    webrtc_streamer(key="sign-detection", video_processor_factory=VideoProcessor)

elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save uploaded file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        st.video("temp_video.mp4")

        # Process Video
        cap = cv2.VideoCapture("temp_video.mp4")
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Predict Sign
            pred_class, confidence = predict_sign(frame)

            # Speak the predicted sign
            cleaned_pred_class = pred_class.replace("_", " ")
            speak_text(cleaned_pred_class)

            # Update detected sign text box
            detected_sign_placeholder.text(f"Detected Sign: {cleaned_pred_class}")

            # Display Prediction on Frame
            cv2.putText(frame, f'{pred_class} ({confidence:.2f}%)', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            stframe.image(frame, channels="BGR")

        cap.release()
