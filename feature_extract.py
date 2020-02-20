import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from scipy import misc
from six.moves import xrange
import cv2
from sklearn.preprocessing import normalize
import glob

def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def preprocess_img(x):
    x = cv2.resize(x, (160, 160))
    # mean = np.mean(x)
    # std = np.std(x)
    # std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    # y = np.multiply(np.subtract(x, mean), 1/std_adj)
    y = (np.float32(x) - 127.5) / 128.0
    return np.expand_dims(y, 0)

parser = argparse.ArgumentParser(description='face identity prediction')
parser.add_argument('--model', default='../model_results/facenet/20180402-114759/20180402-114759.pb', help='path to load model.')
parser.add_argument('--dataset', type=str, default='../datasets/ffhq_128x128/img', help='path to the dataset we want to label')
args = parser.parse_args()

graph = load_pb(args.model)
input = graph.get_tensor_by_name('input:0')
output = graph.get_tensor_by_name('embeddings:0')
phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

id_features=[]
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(graph=graph, config=config) as sess:
    images=sorted(glob.glob(os.path.join(args.dataset,"*.png")))
    for i, file in enumerate(images):
        if i==2000:
            break
        img = cv2.imread(file)
        print(file)
        embed = sess.run(output, feed_dict={input: preprocess_img(img), phase_train_placeholder: False})
        embed=normalize(embed)
        embed=embed.reshape((512,))
        id_features.append(embed)

id_features = np.array(id_features, dtype=np.float32)
print("shape of id_feature is")
print(id_features.shape)
np.save('id_features.npy', id_features)
# pickle.dump(id_features, open(os.path.join(args.dataset, "id_features.p"), "wb"))


