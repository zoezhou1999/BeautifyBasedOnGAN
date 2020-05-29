#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
# import itertools
import os
import csv
import glob
import numpy as np

np.set_printoptions(precision=2)

import openface
import sys
sys.path.append("/Users/zhouyuhongze/torch/install/bin/th")

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'openface/models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

# parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
parser.add_argument('--results_dir', '-results_dir', help='batch image beautification results', default='dean_cond_batch16', type=str)
parser.add_argument('--src_dir', '-src_dir', help='original images path', default='dean_cond_batch16', type=str)
parser.add_argument('--final_iteration', '-final_iteration', help='mark the final beautificaton result', default=572, type=int)
parser.add_argument('--csv_name', '-csv_name', help='csv file name', default='dean_cond_batch16', type=str)

parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))


def getRep(imgPath):
    if args.verbose:
        print("Processing {}.".format(imgPath))
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    bgrImg=cv2.resize(bgrImg,(args.imgDim,args.imgDim))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        print("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace=None
    if bb is not None:
        alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            print("Unable to align image: {}".format(imgPath))
        if args.verbose:
            print("  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    if alignedFace is not None:
        rep = net.forward(alignedFace)
    else:
        print("use original RGB image to extract feature")
        rep = net.forward(rgbImg)
    if args.verbose:
        print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
        print("Representation:")
        print(rep)
        print("-----\n")
    return rep

paths=sorted(glob.glob(os.path.join(args.src_dir,"*.png")))
mean_dis=0
with open(args.csv_name+ ".csv", mode='w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['image_name', 'squared l2 distance'])
    for path in paths:
        name=os.path.basename(path)
        # name=name[0:name.find("_")]
        name=name[0:name.find(".")]

        #These for Beholder-XXXX
        result_path=os.path.join(args.results_dir,str(name))

        # result_path_image=os.path.join(result_path,str(args.final_iteration)+"_0.png")
        result_path_image=os.path.join(result_path,"{}_0.png".format(args.final_iteration))

        #These for InterFaceGAN-XXXX
        # result_path_image=os.path.join(args.results_dir,name+"_0.png")
        # print(path,name,result_path_image)

        print(path,name,result_path,result_path_image)
        # Squared l2 distance between representations
        d = getRep(path)-getRep(result_path_image)
        d_2=np.dot(d, d)
        mean_dis+=d_2
        writer.writerow([name,d_2])

with open(args.csv_name+ ".txt", mode='w') as f:
    mean_dis=mean_dis/len(paths)
    f.writelines("image num: {};\n".format(len(paths)))
    f.writelines("mean_dis: {};\n".format(mean_dis))