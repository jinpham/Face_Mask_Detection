import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
gray = cv2.imread('images/emilia-clarke/1.jpg', 0)
plt.figure(figsize=(12,8))
plt.imshow(gray, cmap='gray')
plt.show()

# Detect faces
faces = face_cascade.detectMultiScale(
gray,
scaleFactor=1.1,
minNeighbors=5,
flags=cv2.CASCADE_SCALE_IMAGE
)
# For each face
for (x, y, w, h) in faces:
    # Draw rectangle around the face
    plt.plot((x,x+w),(y,y+h), 'bD--')
    print((x,y))
    print((w,h))
    print((x+w,y+h))
    cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 2)

plt.figure(figsize=(12,8))
plt.imshow(gray, cmap='gray')
plt.show()