from tsne import tsne
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import sys
import glob
import pylab


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.load('id_features.npy')
    labels1 = np.load("gender.npy")
    labels2 = np.load("eyeglasses.npy")
    Y = tsne(X, 2, 50, 20.0)
    pylab.close('all')
    pylab.ioff()
    fig = pylab.figure(figsize=(10, 10))
    fig.clf()
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels1)
    pylab.title("FaceNet-Gender")
    pylab.savefig("FaceNet-Gender.png")
    pylab.close('all')
    pylab.ioff()
    fig = pylab.figure(figsize=(10, 10))
    fig.clf()
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels2)
    pylab.title("FaceNet-Eyeglasses")
    pylab.savefig("FaceNet-Eyeglasses.png")
