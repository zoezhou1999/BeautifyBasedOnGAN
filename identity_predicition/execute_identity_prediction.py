import face_model
import argparse
import cv2
import sys
import numpy as np
import pickle
import os

parser = argparse.ArgumentParser(description='face identity prediction')

parser.add_argument('--image-size', default='112,112', help='') #no use maybe
parser.add_argument('--model', default='./models/model-r50-am-lfw/model,0000', help='path to load model.')
parser.add_argument('--ga-model', default='./models/gamodel-r50/model,0000', help='path to load model.')
parser.add_argument('--dataset', type=str, default='../datasets/ffhq128x128', help='path to the dataset we want to label')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)

id_features=[]
images_dir = "{0}/img".format(args.dataset)
number_of_images = len(os.listdir(images_dir))
for i, file in enumerate(sorted(os.listdir(images_dir))):
    img = cv2.resize(cv2.imread(os.path.join(images_dir,file)), (112, 112))
    img = model.get_input(img)
    f = model.get_feature(img)
    print(f[0:10])
    id_features.append(f)
id_features=np.array(id_features,dtype=np.float32)
print("shape of id_feature is")
print(id_features.shape)
pickle.dump(id_features, open(os.path.join(args.dataset,"id_features.p"), "wb"))
