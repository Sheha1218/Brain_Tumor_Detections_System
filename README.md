# Brain_Tumor_Detections_System
![Screenshot 2025-04-29 113913](https://github.com/user-attachments/assets/6f2a5342-0c71-4f01-9e2d-19dce4b4218c)


 Brain Tumor Detection Web App
This project is a deep learning-powered web application for classifying brain tumors using MRI images. The system identifies four major tumor types: Glioma, Meningioma, Pituitary, and No Tumor.

The application is built with TensorFlow, OpenCV, Flask, and a fully responsive HTML/CSS/JavaScript frontend. Users can upload an MRI scan image through the browser, and the app performs prediction using a trained CNN model, returning the tumor type and confidence level.

🔍 Key Features
📷 Upload and preview MRI images directly in the browser

🧠 Deep learning-based classification using a custom-trained CNN model

⚙️ Real-time predictions with confidence score

🌐 Fully responsive frontend using modern CSS (glassmorphism, loaders, styled buttons)

☁️ Easily deployable via Azura

🛠️ Tech Stack
Frontend: HTML5, CSS3, JavaScript

Backend: Python, Flask

AI Model: TensorFlow / Keras CNN

Other Tools: OpenCV, NumPy, Git

🧪 Classes Detected
Glioma Tumor

Meningioma Tumor

Pituitary Tumor

No Tumor

🚀 How It Works
The user uploads an MRI image using the web UI.

The image is sent to a Flask backend and preprocessed using OpenCV.

A trained TensorFlow model predicts the class.

The result is returned and displayed along with confidence.
