# Multi-Modal-Emotion-Detection-System
Project of Emotion Detection using CNN in Deep Learning and NLP for text classification

# Real-Time Emotion Detection
This is a real-time emotion detection system that uses a trained deep learning model to classify emotions in live video feed from a webcam. It detects faces in the video frames and applies emotion classification to provide real-time emotion labels.

Requirements
To run this code, you need the following dependencies:

Python 3.x
Keras
TensorFlow
OpenCV (cv2)
NumPy
Installation
Follow the instructions below to install the necessary dependencies:

Install Python 3.x from the official Python website: python.org.

Install Keras and TensorFlow by running the following command:

shell
Copy code
pip install keras tensorflow
Install OpenCV (cv2) by running the following command:

shell
Copy code
pip install opencv-python
Install NumPy by running the following command:

shell
Copy code
pip install numpy
Usage
To run the code, follow the steps below:

Clone or download the code from the repository.

Make sure all the required dependencies are installed as mentioned in the Installation section.

Place the haarcascade_frontalface_default.xml file and the trained model file (model.h5) in the same directory as the code file.

Open a terminal or command prompt and navigate to the directory containing the code.

Run the following command to start the real-time emotion detection:

shell
Copy code
python emotion_detection.py
The webcam will be accessed, and you will see a window titled "Emotion Detector" showing the live video feed with emotion labels drawn on the detected faces.

Press the 'q' key to quit and exit the program.

# Text Classification

Text Preprocessing and Emoji Visualization
This code performs text preprocessing tasks such as removing stopwords, tokenization, lemmatization, and visualizes the frequency of emojis in a given text. It utilizes NLTK, pandas, regex, emoji, Matplotlib, and Gradio libraries.

Requirements
To run this code, you need the following dependencies:

Python 3.x
NLTK
Pandas
Matplotlib
Gradio
Installation
Follow the instructions below to install the necessary dependencies:

Install Python 3.x from the official Python website: python.org.

Install NLTK, Pandas, Matplotlib, and Gradio by running the following command:
pip install nltk pandas matplotlib gradio
