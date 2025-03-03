#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# Define dataset path
data_dir = "./SData"

# Image parameters
img_height, img_width = 128, 128  # Increased resolution for better accuracy
batch_size = 32

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

# Training data
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation data
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Compute class weights (if classes are imbalanced)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# Load MobileNetV2 as base model
base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base layers

# Build the final model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_data.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping & learning rate adjustment
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model
epochs = 50
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[early_stop, lr_scheduler]
)

# Save the trained model
model.save("sign_language_model_transfer.keras")

# Evaluate the model
test_loss, test_acc = model.evaluate(val_data)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


# In[2]:


get_ipython().system('pip install tensorflow')


# In[ ]:


import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

# Load the trained model
model = tf.keras.models.load_model("sign_language_model_transfer.keras")

# Load class indices
class_indices = {v: k for k, v in enumerate(sorted(os.listdir("./SData")))}  # Auto-detect classes
index_to_class = {v: k for k, v in class_indices.items()}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Preprocess Frame Function
def preprocess_frame(frame):
    tensor = tf.image.resize(frame, [128, 128])  # Resize to 128x128
    tensor = tf.cast(tensor, tf.float32) / 255.0  # Normalize
    tensor = tf.expand_dims(tensor, axis=0)      # Add batch dimension
    return tensor

# Initialize Webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for natural webcam view
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Draw Hand Landmarks & Predict Sign
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract ROI for prediction (Use the whole frame or adjust as needed)
            combined_hand_img = rgb_frame

            # Preprocess for model
            combined_hand_img = preprocess_frame(combined_hand_img)

            # Make Prediction
            predictions = model.predict(combined_hand_img)
            pred_index = np.argmax(predictions[0])
            pred_class = index_to_class.get(pred_index, "Unknown")
            confidence = np.max(predictions[0]) * 100

            # Display Prediction
            cv2.putText(frame, f'{pred_class} ({confidence:.2f}%)', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            cv2.putText(frame, "No Hands Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show Frame
        cv2.imshow('SignLoom_Phrases', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release Resources
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




