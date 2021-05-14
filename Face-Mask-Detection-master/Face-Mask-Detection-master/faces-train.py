import cv2
import os
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
# đọc tên tất các folder trong project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# đọc tên tất cả các file trong folder images
image_dir = os.path.join(BASE_DIR, "images")


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
y_labels = []
x_train = []
x_ok = []
y_ok = []
i = 0
for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("JPG") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()
			# print(label,path)
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			print(label_ids)
			#y_labels.append(label) # some number
			#x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
			# chuyển hình ảnh sang màu xám theo kiểu L
			pil_image = Image.open(path).convert("L") # grayscale
			size = (550, 550)
			# thay đổi hình ảnh theo size vừa cho và tiến hành khử răng cưa cho hình ảnh mịn hơn
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			# phân tích hình ảnh thành số từ 0 -> 255 thang màu xám unit8
			image_array = np.array(final_image, "uint8")
			# print(image_array)
			# i+=1
			# print(i)
			# sử dụng bộ lọc để nhận diện khuôn mặt
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.05, minNeighbors=5)
			# plt.plot(faces, 'bD--')
			for (x,y,w,h) in faces:

				print(x)
				print(y)
				print(w)
				print(h)
				# print("    ")
				# x_ok.append(x+w)
				# y_ok.append(y+h)
				roi = image_array[y:y+h, x:x+w]
				# cv2.rectangle(faces, (x, y), (x + w, y + h), (255, 255, 255), 2)
				plt.plot([x,x+w], [y,y+h])
				plt.figure(figsize=(12, 8))
				plt.imshow(faces, cmap='gray')
				plt.show()
				print(image_array[y:y+h, x:x+w])
				x_train.append(roi)
				y_labels.append(id_)

# print(y_labels)
# print(x_train)
# plt.style.use("ggplot")
# plt.figure()
#
# plt.plot(x, 'bD--')
#
# plt.savefig("plot2.png")

with open("pickles/face-labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizers/face-trainner.yml")