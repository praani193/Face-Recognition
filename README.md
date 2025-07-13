# Face-Recognition
Face Recognition – Deep Learning Project
1. Introduction
Face detection is a key technology in computer vision, used in various applications such as security systems, attendance tracking, and human-computer interaction.

This project implements a real-time face detection and recognition system using OpenCV and Tkinter, allowing users to detect faces from live webcam feeds. The system captures and processes facial images, providing an interactive way to identify a person.

With its user-friendly interface, this project serves as a foundation for further advancements in facial analysis and recognition.

2. Dataset Description
2.1 Structure
The dataset consists of facial images captured in real-time via webcam. It is stored as follows:

dataset1/
├── Person_1/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── Person_2/
    ├── 1.jpg
    ├── 2.jpg
    └── ...

Each person has their own subfolder containing 300 images captured during detection.

2.2 Image Specifications
Format: JPEG (.jpg)

Resolution: 130 × 100 pixels

Color Mode: BGR

Captured Using: Webcam

3. Methodology
The system operates in five main stages:

3.1 User Input & Dataset Setup
Prompts user for a name using Tkinter.simpledialog.askstring().

Creates a subfolder under dataset1/ with the entered name.

If the folder exists, new images are added.

3.2 Image Acquisition
Uses cv2.VideoCapture(0) to start webcam feed.

Continuously captures frames using _ , img = cam.read().

3.3 Face Detection & Recognition
Converts frames to grayscale using cv2.cvtColor() for better accuracy.

Uses detectMultiScale() to locate faces.

Draws a green rectangle around detected faces and displays a detection message.

In recognition mode, matches faces against the trained dataset to identify the person.

3.4 Image Processing & Storage
Crops the detected face:

python
Copy
Edit
face_only = img[y:y + h, x:x + w]
Resizes to 130 × 100 using cv2.resize().

Saves the image as 1.jpg, 2.jpg, ..., up to 300.jpg.

Allows early stopping with the Esc key (cv2.waitKey(10) == 27).

3.5 User Interface (GUI)
Created using Tkinter.

Contains:

A title: "Choose an Option"

Two buttons: "Face Detect" and "Face Recognition"

Well-structured layout with padding for visual clarity

4. System Requirements
Operating System: Windows / macOS / Linux

Python Version: 3.7 or higher

Camera: Required for real-time detection

GPU: Optional (recommended for faster performance)
