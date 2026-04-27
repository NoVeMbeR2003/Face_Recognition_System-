🚀 Face Recognition System using OpenCV (LBPH)

A complete Face Recognition System built using Python and OpenCV that can collect datasets, train a model, and recognize faces in real-time using a webcam.

This project demonstrates the core concepts of computer vision and machine learning workflows in a simple and practical way.

📌 Features
🎥 Real-time face detection using Haar Cascade
📂 Automatic dataset creation with user input
🧠 Model training using LBPH algorithm
🔍 Real-time face recognition
🏷️ Displays recognized person's name
❓ Detects unknown faces
⚡ Fast and lightweight (no GPU required)
🧠 How It Works
1️⃣ Dataset Collection
Captures face images using webcam
Stores them in a structured folder (dataset/person_name/)
Converts images to grayscale for consistency
2️⃣ Model Training
Reads all dataset images
Assigns labels to each person
Trains the model using LBPH (Local Binary Pattern Histogram)
Saves trained model (face_model.yml) and label mapping (labels.npy)
3️⃣ Face Recognition
Detects faces in real-time
Compares with trained model
Displays the predicted name on screen
🛠️ Technologies Used
Python
OpenCV (opencv-contrib-python)
NumPy
