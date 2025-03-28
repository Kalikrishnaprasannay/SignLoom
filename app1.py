{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "83ccb248-e141-4476-a915-4473e9a26e87",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-04 10:46:47.230 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-04 10:46:47.242 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-04 10:46:47.243 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-04 10:46:47.243 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-04 10:46:47.243 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-04 10:46:47.245 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-04 10:46:47.248 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-04 10:46:47.252 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-04 10:46:47.253 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-04 10:46:47.255 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-04 10:46:47.257 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-04 10:46:47.258 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-04 10:46:47.260 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import cv2\n",
    "import mediapipe as mp\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import os\n",
    "from PIL import Image\n",
    "\n",
    "# Load the trained model\n",
    "model = tf.keras.models.load_model(\"sign_language_model_transfer.keras\")\n",
    "\n",
    "# Load class indices\n",
    "class_indices = {v: k for k, v in enumerate(sorted(os.listdir(\"./SData\")))}\n",
    "index_to_class = {v: k for k, v in class_indices.items()}\n",
    "\n",
    "# Initialize MediaPipe\n",
    "mp_hands = mp.solutions.hands\n",
    "mp_drawing = mp.solutions.drawing_utils\n",
    "\n",
    "def preprocess_frame(frame):\n",
    "    tensor = tf.image.resize(frame, [128, 128])\n",
    "    tensor = tf.cast(tensor, tf.float32) / 255.0\n",
    "    tensor = tf.expand_dims(tensor, axis=0)\n",
    "    return tensor\n",
    "\n",
    "def predict_sign(frame):\n",
    "    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "    frame_processed = preprocess_frame(frame_rgb)\n",
    "    predictions = model.predict(frame_processed)\n",
    "    pred_index = np.argmax(predictions[0])\n",
    "    pred_class = index_to_class.get(pred_index, \"Unknown\")\n",
    "    confidence = np.max(predictions[0]) * 100\n",
    "    return pred_class, confidence\n",
    "\n",
    "st.title(\"Sign Language Recognition\")\n",
    "\n",
    "option = st.radio(\"Choose input method:\", (\"Upload Video\", \"Use Webcam\"))\n",
    "\n",
    "if option == \"Upload Video\":\n",
    "    uploaded_file = st.file_uploader(\"Upload a .mp4 file\", type=[\"mp4\"])\n",
    "    if uploaded_file is not None:\n",
    "        st.video(uploaded_file)\n",
    "        temp_video_path = \"temp_video.mp4\"\n",
    "        with open(temp_video_path, \"wb\") as f:\n",
    "            f.write(uploaded_file.read())\n",
    "        st.video(temp_video_path)\n",
    "        process_video(temp_video_path)\n",
    "        cap = cv2.VideoCapture(temp_video_path)\n",
    "        \n",
    "        with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:\n",
    "            def process_video(video_path):\n",
    "                cap = cv2.VideoCapture(video_path)\n",
    "                while cap.isOpened():\n",
    "                    ret, frame = cap.read()\n",
    "                    if not ret:\n",
    "                        break\n",
    "                \n",
    "                    pred_class, confidence = predict_sign(frame)\n",
    "                \n",
    "                    cv2.putText(frame, f'{pred_class} ({confidence:.2f}%)', (10, 30),\n",
    "                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)\n",
    "                    st.image(frame, channels=\"BGR\")\n",
    "                cap.release()\n",
    "        cap.release()\n",
    "\n",
    "elif option == \"Use Webcam\":\n",
    "    st.write(\"Press 'Start' to use webcam.\")\n",
    "    start_webcam = st.button(\"Start Webcam\")\n",
    "    \n",
    "    if start_webcam:\n",
    "        cap = cv2.VideoCapture(0)\n",
    "        \n",
    "        with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:\n",
    "            while cap.isOpened():\n",
    "                ret, frame = cap.read()\n",
    "                if not ret:\n",
    "                    break\n",
    "                \n",
    "                frame = cv2.flip(frame, 1)\n",
    "                pred_class, confidence = predict_sign(frame)\n",
    "                \n",
    "                cv2.putText(frame, f'{pred_class} ({confidence:.2f}%)', (10, 30),\n",
    "                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)\n",
    "                st.image(frame, channels=\"BGR\")\n",
    "                \n",
    "                if st.button(\"Stop Webcam\"):\n",
    "                    break\n",
    "        cap.release()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "0c3a2bdd-90a7-4c86-a461-7aa879bd0f47",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.10 (hand_env)",
   "language": "python",
   "name": "hand_env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
