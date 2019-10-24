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

class FaceNet():

    def __init__(self, model_path):
        self.model_path = model_path
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        graph = self.load_pb(self.model_path)
        self.input = graph.get_tensor_by_name('input:0')
        self.output = graph.get_tensor_by_name('embeddings:0')
        self.phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
        self.session.graph = graph

    def load_pb(self, path_to_pb):
        with tf.gfile.GFile(path_to_pb, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            return graph

    def preprocess_img(self, x):
        x = cv2.resize(x, (160, 160))
        y = (np.float32(x) - 127.5) / 128.0
        return np.expand_dims(y, 0)

    def predict(self, img_path):
        img = cv2.imread(img_path)
        embed = self.session.run(self.output, feed_dict={self.input: self.preprocess_img(img), self.phase_train_placeholder: False})
        embed = normalize(embed)
        embed = embed.reshape((512,))
        return embed