# USAGE
# python generate_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image resim.jpg 

import numpy as np
import dlib
import cv2
import math
import imutils
from imutils import face_utils

def predictLandMarks(image):
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	# load the input image, resize it, and convert it to grayscale
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		crop_img = image[y:y+h,x:x+w]
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image 68 coordinates
		landmarks = []
		for (x, y) in shape:
			landmarks.append((x,y))
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	return landmarks