# import bz2
# from keras.utils import get_file



# def unpack_bz2(src_path):
#     data = bz2.BZ2File(src_path).read()
#     dst_path = src_path[:-4]
#     with open(dst_path, 'wb') as fp:
#         fp.write(data)
#     return dst_path

# LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
# landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
#                                                LANDMARKS_MODEL_URL, cache_subdir='temp'))
# print(landmarks_model_path)

from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
vgg16 = VGG16(include_top=False, input_shape=(256, 256, 3))
perceptual_model = Model(vgg16.input, vgg16.layers[9].output)