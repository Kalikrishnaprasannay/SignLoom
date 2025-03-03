
{
"cells":[
0:{
"cell_type":"code"
"execution_count":1
"id":"3a486eb8-f326-42dc-80e8-99663f65abe4"
"metadata":{}
"outputs":[
0:{
"name":"stdout"
"output_type":"stream"
"text":[
0:"Defaulting to user installation because normal site-packages is not writeableNote: you may need to restart the kernel to use updated packages.
"
]
}
1:{
"name":"stderr"
"output_type":"stream"
"text":[
0:"WARNING: Ignoring invalid distribution ~ip (C:\Users\Kalik\AppData\Roaming\Python\Python312\site-packages)
"
1:"WARNING: Ignoring invalid distribution ~ip (C:\Users\Kalik\AppData\Roaming\Python\Python312\site-packages)
"
2:"WARNING: Ignoring invalid distribution ~ip (C:\Users\Kalik\AppData\Roaming\Python\Python312\site-packages)
"
]
}
2:{
"name":"stdout"
"output_type":"stream"
"text":[
0:"
"
1:"Requirement already satisfied: streamlit in c:\anaconda\lib\site-packages (1.32.0)
"
2:"Requirement already satisfied: tensorflow in c:\users\kalik\appdata\roaming\python\python312\site-packages (2.18.0)
"
3:"Requirement already satisfied: pillow in c:\users\kalik\appdata\roaming\python\python312\site-packages (10.3.0)
"
4:"Requirement already satisfied: numpy in c:\anaconda\lib\site-packages (1.26.4)
"
5:"Requirement already satisfied: opencv-python in c:\users\kalik\appdata\roaming\python\python312\site-packages (4.11.0.86)
"
6:"Requirement already satisfied: altair<6,>=4.0 in c:\anaconda\lib\site-packages (from streamlit) (5.0.1)
"
7:"Requirement already satisfied: blinker<2,>=1.0.0 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from streamlit) (1.9.0)
"
8:"Requirement already satisfied: cachetools<6,>=4.0 in c:\anaconda\lib\site-packages (from streamlit) (5.3.3)
"
9:"Requirement already satisfied: click<9,>=7.0 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from streamlit) (8.1.7)
"
10:"Requirement already satisfied: packaging<24,>=16.8 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from streamlit) (23.2)
"
11:"Requirement already satisfied: pandas<3,>=1.3.0 in c:\anaconda\lib\site-packages (from streamlit) (2.2.2)
"
12:"Requirement already satisfied: protobuf<5,>=3.20 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from streamlit) (4.25.6)
"
13:"Requirement already satisfied: pyarrow>=7.0 in c:\anaconda\lib\site-packages (from streamlit) (14.0.2)
"
14:"Requirement already satisfied: requests<3,>=2.27 in c:\anaconda\lib\site-packages (from streamlit) (2.32.2)
"
15:"Requirement already satisfied: rich<14,>=10.14.0 in c:\anaconda\lib\site-packages (from streamlit) (13.3.5)
"
16:"Requirement already satisfied: tenacity<9,>=8.1.0 in c:\anaconda\lib\site-packages (from streamlit) (8.2.2)
"
17:"Requirement already satisfied: toml<2,>=0.10.1 in c:\anaconda\lib\site-packages (from streamlit) (0.10.2)
"
18:"Requirement already satisfied: typing-extensions<5,>=4.3.0 in c:\anaconda\lib\site-packages (from streamlit) (4.11.0)
"
19:"Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in c:\anaconda\lib\site-packages (from streamlit) (3.1.37)
"
20:"Requirement already satisfied: pydeck<1,>=0.8.0b4 in c:\anaconda\lib\site-packages (from streamlit) (0.8.0)
"
21:"Requirement already satisfied: tornado<7,>=6.0.3 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from streamlit) (6.4)
"
22:"Requirement already satisfied: watchdog>=2.1.5 in c:\anaconda\lib\site-packages (from streamlit) (4.0.1)
"
23:"Requirement already satisfied: tensorflow-intel==2.18.0 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorflow) (2.18.0)
"
24:"Requirement already satisfied: absl-py>=1.0.0 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorflow-intel==2.18.0->tensorflow) (2.1.0)
"
25:"Requirement already satisfied: astunparse>=1.6.0 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorflow-intel==2.18.0->tensorflow) (1.6.3)
"
26:"Requirement already satisfied: flatbuffers>=24.3.25 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorflow-intel==2.18.0->tensorflow) (25.2.10)
"
27:"Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorflow-intel==2.18.0->tensorflow) (0.6.0)
"
28:"Requirement already satisfied: google-pasta>=0.1.1 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorflow-intel==2.18.0->tensorflow) (0.2.0)
"
29:"Requirement already satisfied: libclang>=13.0.0 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorflow-intel==2.18.0->tensorflow) (18.1.1)
"
30:"Requirement already satisfied: opt-einsum>=2.3.2 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorflow-intel==2.18.0->tensorflow) (3.4.0)
"
31:"Requirement already satisfied: setuptools in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorflow-intel==2.18.0->tensorflow) (69.0.2)
"
32:"Requirement already satisfied: six>=1.12.0 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorflow-intel==2.18.0->tensorflow) (1.16.0)
"
33:"Requirement already satisfied: termcolor>=1.1.0 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorflow-intel==2.18.0->tensorflow) (2.5.0)
"
34:"Requirement already satisfied: wrapt>=1.11.0 in c:\anaconda\lib\site-packages (from tensorflow-intel==2.18.0->tensorflow) (1.14.1)
"
35:"Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorflow-intel==2.18.0->tensorflow) (1.70.0)
"
36:"Requirement already satisfied: tensorboard<2.19,>=2.18 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorflow-intel==2.18.0->tensorflow) (2.18.0)
"
37:"Requirement already satisfied: keras>=3.5.0 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorflow-intel==2.18.0->tensorflow) (3.8.0)
"
38:"Requirement already satisfied: h5py>=3.11.0 in c:\anaconda\lib\site-packages (from tensorflow-intel==2.18.0->tensorflow) (3.11.0)
"
39:"Requirement already satisfied: ml-dtypes<0.5.0,>=0.4.0 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorflow-intel==2.18.0->tensorflow) (0.4.1)
"
40:"Requirement already satisfied: jinja2 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from altair<6,>=4.0->streamlit) (3.1.4)
"
41:"Requirement already satisfied: jsonschema>=3.0 in c:\anaconda\lib\site-packages (from altair<6,>=4.0->streamlit) (4.19.2)
"
42:"Requirement already satisfied: toolz in c:\anaconda\lib\site-packages (from altair<6,>=4.0->streamlit) (0.12.0)
"
43:"Requirement already satisfied: colorama in c:\users\kalik\appdata\roaming\python\python312\site-packages (from click<9,>=7.0->streamlit) (0.4.6)
"
44:"Requirement already satisfied: gitdb<5,>=4.0.1 in c:\anaconda\lib\site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.7)
"
45:"Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from pandas<3,>=1.3.0->streamlit) (2.8.2)
"
46:"Requirement already satisfied: pytz>=2020.1 in c:\anaconda\lib\site-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)
"
47:"Requirement already satisfied: tzdata>=2022.7 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)
"
48:"Requirement already satisfied: charset-normalizer<4,>=2 in c:\anaconda\lib\site-packages (from requests<3,>=2.27->streamlit) (2.0.4)
"
49:"Requirement already satisfied: idna<4,>=2.5 in c:\anaconda\lib\site-packages (from requests<3,>=2.27->streamlit) (3.7)
"
50:"Requirement already satisfied: urllib3<3,>=1.21.1 in c:\anaconda\lib\site-packages (from requests<3,>=2.27->streamlit) (2.2.2)
"
51:"Requirement already satisfied: certifi>=2017.4.17 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from requests<3,>=2.27->streamlit) (2023.11.17)
"
52:"Requirement already satisfied: markdown-it-py<3.0.0,>=2.2.0 in c:\anaconda\lib\site-packages (from rich<14,>=10.14.0->streamlit) (2.2.0)
"
53:"Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from rich<14,>=10.14.0->streamlit) (2.17.2)
"
54:"Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\anaconda\lib\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.18.0->tensorflow) (0.43.0)
"
55:"Requirement already satisfied: smmap<5,>=3.0.1 in c:\anaconda\lib\site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.0)
"
56:"Requirement already satisfied: MarkupSafe>=2.0 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from jinja2->altair<6,>=4.0->streamlit) (3.0.2)
"
57:"Requirement already satisfied: attrs>=22.2.0 in c:\anaconda\lib\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.1.0)
"
58:"Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\anaconda\lib\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.7.1)
"
59:"Requirement already satisfied: referencing>=0.28.4 in c:\anaconda\lib\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.30.2)
"
60:"Requirement already satisfied: rpds-py>=0.7.1 in c:\anaconda\lib\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.10.6)
"
61:"Requirement already satisfied: namex in c:\users\kalik\appdata\roaming\python\python312\site-packages (from keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow) (0.0.8)
"
62:"Requirement already satisfied: optree in c:\users\kalik\appdata\roaming\python\python312\site-packages (from keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow) (0.14.0)
"
63:"Requirement already satisfied: mdurl~=0.1 in c:\anaconda\lib\site-packages (from markdown-it-py<3.0.0,>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.0)
"
64:"Requirement already satisfied: markdown>=2.6.8 in c:\anaconda\lib\site-packages (from tensorboard<2.19,>=2.18->tensorflow-intel==2.18.0->tensorflow) (3.4.1)
"
65:"Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorboard<2.19,>=2.18->tensorflow-intel==2.18.0->tensorflow) (0.7.2)
"
66:"Requirement already satisfied: werkzeug>=1.0.1 in c:\users\kalik\appdata\roaming\python\python312\site-packages (from tensorboard<2.19,>=2.18->tensorflow-intel==2.18.0->tensorflow) (3.1.3)
"
]
}
]
"source":[
0:"pip install streamlit tensorflow pillow numpy opencv-python"
]
}
1:{
"cell_type":"code"
"execution_count":2
"id":"3cc3a5c7-982b-401f-b3ba-f7279521185f"
"metadata":{}
"outputs":[
0:{
"name":"stderr"
"output_type":"stream"
"text":[
0:"2025-03-03 10:55:20.505 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
"
1:"2025-03-03 10:55:20.505 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
"
2:"2025-03-03 10:55:20.505 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
"
3:"2025-03-03 10:55:20.521 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
"
4:"2025-03-03 10:55:20.523 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
"
5:"2025-03-03 10:55:20.525 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
"
6:"2025-03-03 10:55:20.525 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
"
7:"2025-03-03 10:55:20.527 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
"
8:"2025-03-03 10:55:20.528 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
"
9:"2025-03-03 10:55:20.529 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
"
10:"2025-03-03 10:55:20.532 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
"
11:"2025-03-03 10:55:20.552 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
"
12:"2025-03-03 10:55:20.552 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
"
13:"2025-03-03 10:55:20.552 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
"
14:"2025-03-03 10:55:20.552 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
"
]
}
]
"source":[
0:"import streamlit as st
"
1:"import cv2
"
2:"import numpy as np
"
3:"import tensorflow as tf
"
4:"import mediapipe as mp
"
5:"import os
"
6:"from PIL import Image
"
7:"
"
8:"# Load the trained model
"
9:"model = tf.keras.models.load_model("sign_language_model_transfer.keras")
"
10:"
"
11:"# Load class indices
"
12:"data_dir = "./SData"
"
13:"class_indices = {v: k for k, v in enumerate(sorted(os.listdir(data_dir)))}  # Auto-detect classes
"
14:"index_to_class = {v: k for k, v in class_indices.items()}
"
15:"
"
16:"# Initialize MediaPipe Hands
"
17:"mp_hands = mp.solutions.hands
"
18:"mp_drawing = mp.solutions.drawing_utils
"
19:"
"
20:"# Function to preprocess image
"
21:"def preprocess_frame(frame):
"
22:"    tensor = tf.image.resize(frame, [128, 128])  # Resize to 128x128
"
23:"    tensor = tf.cast(tensor, tf.float32) / 255.0  # Normalize
"
24:"    tensor = tf.expand_dims(tensor, axis=0)      # Add batch dimension
"
25:"    return tensor
"
26:"
"
27:"# Streamlit UI
"
28:"st.title("SignLoom")
"
29:"st.sidebar.header("Options")
"
30:"option = st.sidebar.radio("Choose an option:", ("Upload Image", "Use Webcam"))
"
31:"
"
32:"if option == "Upload Image":
"
33:"    uploaded_file = st.file_uploader("Upload an image of a hand sign", type=["jpg", "png", "jpeg"])
"
34:"    if uploaded_file is not None:
"
35:"        image = Image.open(uploaded_file)
"
36:"        st.image(image, caption="Uploaded Image", use_column_width=True)
"
37:"        
"
38:"        # Convert image for processing
"
39:"        image = np.array(image)
"
40:"        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
"
41:"        image = preprocess_frame(image)
"
42:"        
"
43:"        # Make prediction
"
44:"        predictions = model.predict(image)
"
45:"        pred_index = np.argmax(predictions[0])
"
46:"        pred_class = index_to_class.get(pred_index, "Unknown")
"
47:"        confidence = np.max(predictions[0]) * 100
"
48:"        
"
49:"        st.write(f"**Prediction:** {pred_class} ({confidence:.2f}%)")
"
50:"
"
51:"elif option == "Use Webcam":
"
52:"    st.write("Press 'Start' to access your webcam and detect hand signs.")
"
53:"    run_webcam = st.button("Start Webcam")
"
54:"    
"
55:"    if run_webcam:
"
56:"        cap = cv2.VideoCapture(0)
"
57:"        stframe = st.empty()
"
58:"
"
59:"        with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:
"
60:"            while True:
"
61:"                ret, frame = cap.read()
"
62:"                if not ret:
"
63:"                    st.write("Could not access webcam.")
"
64:"                    break
"
65:"                
"
66:"                frame = cv2.flip(frame, 1)
"
67:"                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
"
68:"                result = hands.process(rgb_frame)
"
69:"                
"
70:"                if result.multi_hand_landmarks:
"
71:"                    for hand_landmarks in result.multi_hand_landmarks:
"
72:"                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
"
73:"                    
"
74:"                    # Preprocess for model
"
75:"                    processed_frame = preprocess_frame(rgb_frame)
"
76:"                    predictions = model.predict(processed_frame)
"
77:"                    pred_index = np.argmax(predictions[0])
"
78:"                    pred_class = index_to_class.get(pred_index, "Unknown")
"
79:"                    confidence = np.max(predictions[0]) * 100
"
80:"                    
"
81:"                    cv2.putText(frame, f'{pred_class} ({confidence:.2f}%)', (10, 30),
"
82:"                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
"
83:"                else:
"
84:"                    cv2.putText(frame, "No Hands Detected", (10, 30),
"
85:"                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
"
86:"                
"
87:"                stframe.image(frame, channels="BGR")
"
88:"                
"
89:"                if st.button("Stop Webcam"):
"
90:"                    break
"
91:"        
"
92:"        cap.release()
"
]
}
2:{
"cell_type":"code"
"execution_count":NULL
"id":"e06553a6-a0f5-4cb0-90a8-0a845c682fa8"
"metadata":{}
"outputs":[]
"source":[]
}
3:{
"cell_type":"code"
"execution_count":NULL
"id":"aae33d7f-732d-4902-9649-6bca63d91593"
"metadata":{}
"outputs":[]
"source":[]
}
]
"metadata":{
"kernelspec":{
"display_name":"Python 3.10 (hand_env)"
"language":"python"
"name":"hand_env"
}
"language_info":{
"codemirror_mode":{
"name":"ipython"
"version":3
}
"file_extension":".py"
"mimetype":"text/x-python"
"name":"python"
"nbconvert_exporter":"python"
"pygments_lexer":"ipython3"
"version":"3.11.0"
}
}
"nbformat":4
"nbformat_minor":5
}
