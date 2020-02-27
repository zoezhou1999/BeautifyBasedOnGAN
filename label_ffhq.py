import numpy
from matplotlib import pyplot as plt
import glob
import os
import numpy as np
from PIL import Image
import subprocess

def tell():
    if os.path.exists('eyeglasses.npy') and os.path.isfile('eyeglasses.npy'):
        subprocess.check_output(['bash', '-c', "rm eyeglasses.npy"])
            
if __name__ == '__main__':
    x = [1, 2, 3]
    labels=[]
    folderpath="./datasets/ffhq_128x128"
    images=sorted(glob.glob(os.path.join(folderpath,"*.png")))
    images_copy=images.copy()
    length=len(images)
    start=0
    if os.path.exists('eyeglasses.npy') and os.path.isfile('eyeglasses.npy'):
        tmp=np.load('eyeglasses.npy')
        start=tmp.shape[0]
        print("original stored num is "+str(start))
        labels=list(tmp.reshape((-1,)))
    plt.ion() # turn on interactive mode
    for i in range(start,len(images)):
        img=Image.open(images[i])
        plt.imshow(img)
        label=None
        print("input eyeglasses label to continue for image "+str(i)+" "+os.path.basename(images[i]))
        while True:
            label= int(input())
            if label==1 or label==2:
                break
            elif label==3:
                if i>0:
                    labels.pop()
                    img_=Image.open(images_copy[i-1])
                    plt.imshow(img_)
                    print("undo previous image! "+images_copy[i-1])
                    print("input eyeglasses label to continue for image "+str(i-1)+" "+os.path.basename(images_copy[i-1]))
                    while True:
                        label= int(input())
                        if label==1 or label==2:
                            break
                        else:
                            print("input correct eyeglasses label to continue!")
                    if label==1 or label==2:
                        labels.append(label)
                        tell()
                        np.save('eyeglasses.npy', (np.array(labels)).reshape((-1,1)))
                    plt.close()
                else:
                    exit(1)
            else:
                print("input correct eyeglasses label to continue!")
        if label==1 or label==2:
            labels.append(label)
            tell()
            np.save('eyeglasses.npy', (np.array(labels)).reshape((-1,1)))
        plt.close()
    labels=np.array(labels)
    labels=labels.reshape((-1,1))
    tell()
    np.save('eyeglasses.npy', labels)
    #1 for female; 2 for male
    #1 for no eyeglasses; 2 for eyeglasses

