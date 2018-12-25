# USAGE
# python generate_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image resim.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import itertools
import math
from gabor_features import gabor_filter
from edge_density import canny_edge_density
from landmark_features import predictLandMarks

def cropFace(image):

	detector = dlib.get_frontal_face_detector()
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	(x, y, w, h) = face_utils.rect_to_bb(rects[0])
	crop_img = image[y:y+h,x:x+w]
	#cv2.imshow("s", crop_img)
	return crop_img

def calculateDistance(pts,landmarkCords):
	x1,y1 = landmarkCords[pts[0]]
	x2,y2 = landmarkCords[pts[1]]
	distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
	return distance

def calculateRatio(pts,landmarkCords):
	x1,y1 = landmarkCords[pts[0]]
	x2,y2 = landmarkCords[pts[1]]
	x3,y3 = landmarkCords[pts[2]]
	x4,y4 = landmarkCords[pts[3]]
	firstDistance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
	secondDistance = math.sqrt((x3-x4)**2 + (y3-y4)**2)
	return firstDistance/secondDistance

def generateSymmetryFeatures(landmarkCords):
	attractivePoints = [0, 4, 8, 12, 16, 17, 21, 22, 26,27, 31, 35, 36, 39, 42, 45,  48, 51, 54, 57]
	combinations = itertools.combinations(attractivePoints, 4)
	ratios = []
	#comb_list = list(combinations)
	#print(comb_list[3264], comb_list[561], comb_list[2498], comb_list[528], comb_list[544]) # most attrracive 5 points
	for combination in combinations:
		ratios.append(calculateRatio(combination,landmarkCords))
	return ratios

def generateDistanceFeatures(landmarkCords):
	distancePoints = []
	for i in range(68):
		distancePoints.append(i)
	combinations = itertools.combinations(distancePoints, 2)
	distances = []
	#comb_list = list(combinations)
	#print (comb_list[1477],comb_list[1476],comb_list[2226],comb_list[1004],comb_list[307]) # most attractive 5 points
	for combination in combinations:
		distances.append(calculateDistance(combination,landmarkCords))
	return distances


if __name__ == '__main__':
	f = open('features.txt','w')
	for i in range (1,4): # for each image change to 500 after you download dataset
		image_path = "data\SCUT-FBP-"+str(i) + ".jpg"
		image = cv2.imread(image_path)
		face = cropFace(image) # get only face using face detector
		landmarkCords = predictLandMarks(image)
		features = [] # all features to be appended

		# generate landmark features
		ratios = generateSymmetryFeatures(landmarkCords)
		features.extend(ratios)

		distances = generateDistanceFeatures(landmarkCords)
		features.extend(distances)

		# apply gabor_filter and get gabor_features from face
		gabor_features = gabor_filter(face)
		features.extend(gabor_features)

		# apply canny edge detecter get edge density feature from face
		edge_density_feature = canny_edge_density(face)
		features.append(edge_density_feature)
		print (len(features))

		f.write(','.join(str("{0:.5f}".format(i)) for i in features))
		f.write("\n")
	#cv2.waitKey(0)