import os
import misc
import numpy as np
import pdb
from config import EasyDict
import tfutil
import argparse
import csv
import tensorflow as tf
import tensorflow_hub as hub
import PIL
from PIL import Image
# import matplotlib.pyplot as plt
from identity_prediction import facenet
from beauty_prediction import beautyrater

# initialize parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', '-results_dir', help='name of training experiment folder', default='dean_cond_batch16', type=str)
parser.add_argument('--labels_size', '-labels_size', help='size of labels vector', default=60, type=int)
parser.add_argument('--iters', '-iters', help='learning rate of algorithm', default=100000, type=int)
parser.add_argument('--lr', '-lr', help='learning rate of algorithm', default=0.1, type=float)
parser.add_argument('--alpha', '-alpha', help='weight of normal loss in relation to vgg loss', default=0.4, type=float)
parser.add_argument('--gpu', '-gpu', help='gpu index for the algorithm to run on', default='0', type=str)
parser.add_argument('--image_path', '-image_path', help='full path to image', default='../datasets/CelebA-HQ/img/03134.png', type=str)
parser.add_argument('--resolution', '-resolution', help='resolution of the generated image', default=128, type=int)

args = parser.parse_args()

# manual parameters
result_subdir = misc.create_result_subdir('results', 'inference_test')
misc.init_output_logging()

# initialize TensorFlow
print('Initializing TensorFlow...')
env = EasyDict()  # Environment variables, set by the main program in train.py.
env.TF_CPP_MIN_LOG_LEVEL = '1'  # Print warnings and errors, but disable debug info.
env.CUDA_VISIBLE_DEVICES = args.gpu  # Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use. change to '0' if first GPU is better
os.environ.update(env)
tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
tf_config['graph_options.place_pruned_graph'] = True  # False (default) = Check that all ops are available on the designated device.
tf_config['gpu_options.allow_growth'] = True
tfutil.init_tf(tf_config)

# load network
network_pkl = misc.locate_network_pkl(args.results_dir)
print('Loading network from "%s"...' % network_pkl)
G, D, Gs = misc.load_network_pkl(args.results_dir, None)

# initiate random input
latents = misc.random_latents(1, Gs, random_state=np.random.RandomState(800))
# labels = np.random.rand(1, args.labels_size+512)

model1=beautyrater.BeautyRater('./beauty_prediction/trained_model/VGG16_beauty_rates-new.pt')
features1=model1.predict(args.image_path)

model2=facenet.FaceNet('./identity_prediction/models/20180402-114759/20180402-114759.pb')
features2=model2.singlePredict(args.image_path)
labels=np.hstack([features1,features2]).astype(np.float32)

# upload image and convert to input tensor
img = PIL.Image.open(args.image_path)
img = img.resize((args.resolution,args.resolution), Image.ANTIALIAS)
img.save((args.image_path).split('/')[-1]) # save image for debug purposes
img = np.asarray(img)
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, axis=0)
img = (img / 127.5) - 1.0 # normalization

# execute algorithm
history = Gs.reverse_gan_for_etalons(latents, labels, img, results_dir=args.results_dir, dest_dir=result_subdir, iters=args.iters, learning_rate=args.lr, alpha=args.alpha)

# save history of latents
with open(result_subdir+'/history_of_latents.txt', 'w') as f:
    for item in history:
        f.write("{}\n".format(item))
        f.write("\n")
