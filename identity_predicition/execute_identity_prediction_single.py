import face_model
import argparse
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--imagepath', default='Tom_Hanks_54745.png', help='')
parser.add_argument('--model', default='./models/model-r50-am-lfw/model,0000', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)

img = cv2.imread(args.imagepath)
img=cv2.resize(img,(112,112))
img = model.get_input(img)
f1 = model.get_feature(img)
print(f1[0:10])