#coding=utf8
import cv2
import dlib
import numpy as np
import sys
import scipy.misc
import scipy.ndimage as ndimage

def get_Landmarks(image_name):
    root = '/home/deanir/classify_beauty_rate-master/'
    PREDICTOR_PATH = root + "data/shape_predictor_68_face_landmarks.dat"
    NUM_LANDMARKS = 68

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)


    path = "../image/"
    im = cv2.imread(path + image_name)
    im_size = im.shape
    print("im_size: {}".format(im_size))

    rects = detector(im,1)

    if len(rects) >= 1:
        print("{} faces detected".format(len(rects)))
    if len(rects) == 0:
        raise NoFaces

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x,p.y] for p in predictor(im,rects[i]).parts()])

    binary_image = np.zeros((im_size[0],im_size[1]))
    for tuple in landmarks:
        y,x = tuple[0,0], tuple[0,1]
        if(x >= 255 or y>= 255):
            print("({},{})".format(x,y))
        else:
            binary_image[x,y] = 255
    scipy.misc.imsave(image_name.split(".")[0] + '_landmarks_.jpg', binary_image)

    return binary_image

def get_facial_outlines(binary_image,color = (255, 0, 0)):

    binary_lines = connect_lines_in_range(0, 16, landmarks, binary_image,color)
    binary_lines = connect_lines_in_range(17, 21, landmarks, binary_lines, color)
    binary_lines = connect_lines_in_range(22, 26, landmarks, binary_lines, color)
    binary_lines = connect_lines_in_range(27, 30, landmarks, binary_lines, color)
    binary_lines = connect_lines_in_range(31, 35, landmarks, binary_lines, color)
    binary_lines = connect_lines_in_range(36, 41, landmarks, binary_lines, color)
    ##close circle
    binary_lines = cv2.line(binary_lines, (landmarks[36][0, 0], landmarks[36][0, 1]), (landmarks[41][0, 0], landmarks[41][0, 1]), color, thickness=1,
                            lineType=8, shift=0)
    binary_lines = connect_lines_in_range(42, 47, landmarks, binary_lines, color)
    ##close circle
    binary_lines = cv2.line(binary_lines, (landmarks[42][0, 0], landmarks[42][0, 1]),
                            (landmarks[47][0, 0], landmarks[47][0, 1]), color, thickness=1,
                            lineType=8, shift=0)
    binary_lines = connect_lines_in_range(48, 54, landmarks, binary_lines, color)
    binary_lines = connect_lines_in_range(55, 59, landmarks, binary_lines, color)
    binary_lines = connect_lines_in_range(60, 67, landmarks, binary_lines, color)
    ##close circle
    binary_lines = cv2.line(binary_lines, (landmarks[60][0, 0], landmarks[60][0, 1]),
                            (landmarks[67][0, 0], landmarks[67][0, 1]), color, thickness=1,
                            lineType=8, shift=0)



    scipy.misc.imsave(image_name.split(".")[0] + '_binary_lines.jpg', binary_lines)
    return binary_lines

def connect_lines_in_range(start,end, landmarks,input_image,color):
    for i in range(start,end):
        pt = landmarks[i]
        pt_next = landmarks[i+1]
        binary_lines = cv2.line(input_image, (pt[0, 0], pt[0, 1]), (pt_next[0, 0], pt_next[0, 1]), color, thickness=1,
                                lineType=8, shift=0)
    return binary_lines
def main():
    image_name = "nir.jpg"
    get_Landmarks(image_name)

if __name__=="__main__":
    main()