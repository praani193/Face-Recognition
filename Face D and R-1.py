import tkinter as tk
from tkinter import simpledialog, messagebox
import os
import cv2
import numpy as np
from deepface import DeepFace
import tempfile
import datetime

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def face_detect():
    try:
        folder_name = simpledialog.askstring("Input", "Enter the folder name:")
        if not folder_name:
            messagebox.showwarning("Input Error", "Folder name cannot be empty!")
            return

        dataset_path = os.path.join("dataset1", folder_name)
        if os.path.exists(dataset_path):
            res = messagebox.askyesno("Folder Exists", "Folder already exists. Add new images?")
            if not res:
                return
        os.makedirs(dataset_path, exist_ok=True)

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Camera Error", "Unable to access webcam.")
            return

        count = 1
        while count <= 300:
            ret, img = cam.read()
            if not ret:
                break

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haar_cascade.detectMultiScale(gray_img, 1.3, 4)
            if len(faces) == 0:
                continue

            for (x, y, w, h) in faces:
                face_crop = img[y:y + h, x:x + w]
                face_resized = cv2.resize(face_crop, (160, 160))
                save_path = os.path.join(dataset_path, f"{count}.jpg")
                cv2.imwrite(save_path, face_resized)
                count += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, "Person Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Face Detection", img)
            if cv2.waitKey(1) == 27:
                break

        cam.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error [Detect]: {str(e)}")

def face_recognize():
    try:
        dataset_path = "dataset1"
        if not os.path.exists(dataset_path):
            messagebox.showerror("Dataset Error", "Dataset folder does not exist!")
            return

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Camera Error", "Unable to access webcam.")
            return

        while True:
            ret, img = cam.read()
            if not ret:
                break

            faces = haar_cascade.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in faces:
                face_crop = img[y:y + h, x:x + w]
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
                    temp_path = temp_img.name
                    cv2.imwrite(temp_path, face_crop)

                try:
                    result = DeepFace.find(img_path=temp_path, db_path=dataset_path, model_name="Facenet512")
                    if len(result) > 0 and not result[0].empty:
                        identity_path = result[0]['identity'][0]
                        if os.path.exists(identity_path):
                            folder_name = identity_path.split(os.sep)[-2]
                            name_text = folder_name
                            color = (0, 255, 0)
                        else:
                            name_text = "Unknown"
                            color = (0, 0, 255)
                    else:
                        name_text = "Unknown"
                        color = (0, 0, 255)
                except Exception as e:
                    print(f"DeepFace error: {e}")
                    name_text = "Unknown"
                    color = (0, 0, 255)

                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    print(f"Temp file delete error: {e}")

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, name_text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            fps = cam.get(cv2.CAP_PROP_FPS)
            cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Face Recognition", img)
            if cv2.waitKey(1) == 27:
                break

        cam.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error [Recognize]: {str(e)}")

root = tk.Tk()
root.title("Face Detection & Recognition")
root.geometry("400x200")

tk.Label(root, text="Choose an Option", font=("Arial", 16)).pack(pady=20)
tk.Button(root, text="Face Detect", command=face_detect, width=20, height=2).pack(pady=10)
tk.Button(root, text="Face Recognition", command=face_recognize, width=20, height=2).pack(pady=10)

root.mainloop()
