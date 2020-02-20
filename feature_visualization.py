from tsne import tsne
import numpy as np
import pylab
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import os
import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys
import glob
import collections
from matplotlib.image import NonUniformImage
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch

if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.load('id_features.npy')
    labels1 = np.load("gender.npy")
    labels2 = np.load("eyeglasses.npy")
    Y = tsne(X, 2, 50, 20.0)
    plt.close('all')
    plt.ioff()
    fig = plt.figure(figsize=(10, 10))
    fig.clf()
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels1)
    plt.title("FaceNet-Gender")
    plt.savefig("FaceNet-Gender.png")
    plt.close('all')
    plt.ioff()
    fig = plt.figure(figsize=(10, 10))
    fig.clf()
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels2)
    plt.title("FaceNet-Eyeglasses")
    plt.savefig("FaceNet-Eyeglasses.png")
