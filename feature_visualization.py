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
    labels1 = (np.load("gender.npy")).reshape((-1,))
    labels2 = (np.load("eyeglasses.npy")).reshape((-1,))
    Y = tsne(X, 2, 50, 20.0)
    colors = ['b','r','g']
    plt.close('all')
    plt.ioff()
    fig = plt.figure(figsize=(10, 10))
    fig.clf()

    for x,y,label in zip(Y[:, 0], Y[:, 1],labels1):
        if label==1:
            labelname="Female"
        else:
            labelname="Male"
        plt.scatter(x,y,color=colors[label],label=labelname, alpha=0.5)
    plt.title("FaceNet-Gender")
    plt.savefig("FaceNet-Gender.png")
    
    plt.close('all')
    plt.ioff()
    fig = plt.figure(figsize=(10, 10))
    fig.clf()
    for x,y,label in zip(Y[:, 0], Y[:, 1],labels2):
        if label==1:
            labelname="W/O Eyeglasses"
        else:
            labelname="With Eyeglasses"
        plt.scatter(x,y,color=colors[label],label=labelname, alpha=0.5)
    plt.title("FaceNet-Eyeglasses")
    plt.savefig("FaceNet-Eyeglasses.png")
