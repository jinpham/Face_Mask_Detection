# import the necessary packages
# import RPi.GPIO as GPIO
# from gpiozero
import winsound
from playsound import playsound
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from threading import Thread
from subprocess import call
import numpy as np
import imutils
import time
import cv2
import pyttsx3
import os
import pickle
from datetime import date
import pymysql.cursors
from datetime import datetime
import Edit


if __name__ == '__main__':
    import Edit

check = False
engine = pyttsx3.init()  # object creation
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}


def alarm():

	# print('call')
	# a = 0
	# s = 'espeak "' + msg + '"'
	# os.system(s)
	if check:
		voices = engine.getProperty('voices')  # getting details of current voice
		# engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
		engine.setProperty('voice', voices[0].id)  # changing index, changes voices. 1 for female
		engine.setProperty('rate', 400)
		engine.say("No Mask")
		# engine.say('My current speaking rate is Tu')
		engine.runAndWait()
		engine.stop()
		# filename = 'tingtingmav.wav'
		# winsound.PlaySound(filename, winsound.SND_FILENAME)
		# playsound('tingtingmav.wav')
		# engine = pyttsx3.init()
		# # engine.say("")
		# # engine.runAndWait()
		# voices = engine.getProperty('voices')
		# engine.setProperty('voice', voices[1].id)
		# engine.setProperty('rate', 400)
		# engine.say("No Mask")
		# engine.runAndWait()
		# engine.stop()

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	#  lấy kích thước của khung và sau đó tạo một đốm màu từ nó
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# cap = cv2.VideoCapture(0)
# loop over the frames from the video stream
while True:
	label = ""
	check = False
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	# ret, frame = vs.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		# determine the class label and color we'll use to draw
		# the bounding box and text
		if mask > withoutMask:
			label = "mask"
			color = (0, 255, 0)
			pass
		else:
			color = (0, 0, 255)
			temp = False
			for (x, y, w, h) in faces:
				roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
				roi_color = frame[y:y + h, x:x + w]
				id_, conf = recognizer.predict(roi_gray)
				if conf >= 4 and conf <= 85:
					check = True
					temp = True
					# recognize? deep learned model predict keras tensorflow pytorch scikit learn
					id_, conf = recognizer.predict(roi_gray)
					label = labels[id_]
					Edit.insert(label)
					# winsound.Beep(1000, 100)
					t = Thread(target=alarm)
					# t.deamon = True
					t.start()
					# engine = pyttsx3.init()
					# voices = engine.getProperty('voices')
					# engine.setProperty('voice', voices[1].id)
					# engine.setProperty('rate', 400)
					# engine.say("No mask")
					# engine.runAndWait()
					# winsound.Beep(1000, 100)
					# filename = 'tingtingmav.wav'
					# winsound.PlaySound(filename, winsound.SND_FILENAME)
					# playsound('tingtingmav.wav')
					# pass
				# else :
				# 	check = True
				# 	label = "No name"
				# 	t = Thread(target=alarm)
				# 	t.deamon = True
				# 	t.start()
				# 	check = False
					# t = Thread(target=alarm, args=('No mask',))
					# t.deamon = True
					# t.start()
					# t.deamon = True
					# t.start()
					# engine = pyttsx3.init()
					# voices = engine.getProperty('voices')
					# engine.setProperty('voice', voices[1].id)
					# engine.setProperty('rate', 400)
					# engine.say("No mask")
					# engine.runAndWait()
					# winsound.Beep(1000, 100)
					# filename = 'tingtingmav.wav'
					# winsound.PlaySound(filename, winsound.SND_FILENAME)
					# playsound('tingtingmav.wav')
					# pass
			if temp is False :
				label = "No Name"
				check = True
				# filename = 'tingtingmav.wav'
				# winsound.PlaySound(filename, winsound.SND_FILENAME)
				# playsound('tingtingmav.wav')
				# winsound.Beep(1000, 100)
				t = Thread(target=alarm)
				# t.deamon = True
				t.start()
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	temp = True
	check = False
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()