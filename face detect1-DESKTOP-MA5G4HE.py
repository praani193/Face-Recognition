import cv2
import os

dataset = "dataset"
name = "champ"

path = os.path.join(dataset, name)
print(os.path.isdir(path))
if not os.path.exists(path):
    os.makedirs(path)

(width, height) = (130, 100)
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

count = 1

while count < 31:
    _, img = cam.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(gray_img, 1.3, 4)

    if len(face) > 0:
        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(count)
            print("Person Detected")
        cv2.putText(img, 'Person Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, 'No Person Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        print("NO Person Detected")
    cv2.imshow("FaceDetection", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    if len(face) > 0:
        for (x, y, w, h) in face:
            face_only = gray_img[y:y + h, x:x + w]
            resize_img = cv2.resize(face_only, (width, height))
            cv2.imwrite("%s/%s.jpg" % (path, count), resize_img)
            count += 1
print("Completed Face Detection")
cam.release()
cv2.destroyAllWindows()
