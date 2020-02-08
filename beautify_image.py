import os
import misc
import numpy as np
import pdb
from config import EasyDict,cache_dir
import tfutil
import argparse
import csv
import tensorflow as tf
import tensorflow_hub as hub
import PIL
from PIL import Image
# import matplotlib.pyplot as plt
import sys
import bz2
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
import multiprocessing
import pickle
from tqdm import tqdm
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel, load_images
from keras.models import load_model
import gc
from beauty_prediction import beautyrater
from identity_prediction import facenet

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


# initialize parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', '-results_dir', help='name of training experiment folder', default='dean_cond_batch16', type=str)
parser.add_argument('--labels_size', '-labels_size', help='size of labels vector', default=572, type=int)
parser.add_argument('--alpha', '-alpha', help='weight of normal loss in relation to vgg loss', default=0.7, type=float)
parser.add_argument('--gpu', '-gpu', help='gpu index for the algorithm to run on', default='0', type=str)
parser.add_argument('--image_path', '-image_path', help='full path to image', default='../datasets/CelebA-HQ/img/03134.png', type=str)
parser.add_argument('--resolution', '-resolution', help='resolution of the generated image', default=128, type=int)
parser.add_argument('--aligned_dir', help='Directory for storing aligned images',default="beautify_image_alighed")
parser.add_argument('--x_scale', default=1, help='Scaling factor for x dimension', type=float)
parser.add_argument('--y_scale', default=1, help='Scaling factor for y dimension', type=float)
parser.add_argument('--em_scale', default=0.1, help='Scaling factor for eye-mouth distance', type=float)
parser.add_argument('--use_alpha', default=False, help='Add an alpha channel for masking', type=bool)
parser.add_argument('--iterations_to_save', default=500, help='iterations_to_save', type=int)

# parser.add_argument('src_dir', help='Directory with images for encoding')
parser.add_argument('--generated_images_dir', help='Directory for storing generated images', default="generated_images")
parser.add_argument('--dlatent_dir', help='Directory for storing dlatent representations', default="latent_representations")
parser.add_argument('--dlabel_dir', help='Directory for storing dlatent representations', default="label_representations")
parser.add_argument('--data_dir', default='data', help='Directory for storing optional models')
parser.add_argument('--mask_dir', default='masks', help='Directory for storing optional masks')
parser.add_argument('--load_last', default='', help='Start with embeddings from directory')
parser.add_argument('--landmarks_model_path', help='Fetch a fl model', default='data/shape_predictor_68_face_landmarks.dat')

# Perceptual model params
parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
parser.add_argument('--resnet_image_size', default=256, help='Size of images for the Resnet model', type=int)
parser.add_argument('--lr', default=0.01, help='Learning rate for perceptual model', type=float)
parser.add_argument('--decay_rate', default=0.8, help='Decay rate for learning rate', type=float)
parser.add_argument('--iterations', default=2500, help='Number of optimization steps for each batch', type=int)
parser.add_argument('--decay_steps', default=10, help='Decay steps for learning rate decay (as a percent of iterations)', type=float)
parser.add_argument('--load_effnet', default='data/finetuned_effnet.h5', help='Model to load for EfficientNet approximation of dlatents')
parser.add_argument('--load_resnet', default='data/finetuned_resnet.h5', help='Model to load for ResNet approximation of dlatents')
parser.add_argument('--load_perc_model', default='data/vgg16_zhang_perceptual.pkl', help='Model to load for ResNet approximation of dlatents')
parser.add_argument('--load_vgg_model', default='data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', help='Model to load for VGG16')
parser.add_argument('--load_vgg_beauty_rater_model', default='../beauty_prediction/trained_model/VGG16_beauty_rates-new.pt', help='Model to load for VGG16')
parser.add_argument('--load_facenet_model', default='./identity_prediction/models/20180402-114759/20180402-114759.pb', help='Model to load for VGG16')

# Loss function options
parser.add_argument('--use_vgg_loss', default=0.4, help='Use VGG perceptual loss; 0 to disable, > 0 to scale.', type=float)
parser.add_argument('--use_vgg_layer', default=9, help='Pick which VGG layer to use.', type=int)
parser.add_argument('--use_pixel_loss', default=1.5, help='Use logcosh image pixel loss; 0 to disable, > 0 to scale.', type=float)
parser.add_argument('--use_mssim_loss', default=100, help='Use MS-SIM perceptual loss; 0 to disable, > 0 to scale.', type=float)
parser.add_argument('--use_lpips_loss', default=100, help='Use LPIPS perceptual loss; 0 to disable, > 0 to scale.', type=float)
# parser.add_argument('--use_l1_penalty', default=1, help='Use L1 penalty on latents; 0 to disable, > 0 to scale.', type=float)
parser.add_argument('--use_beauty_score_loss', default=100, help='Use LPIPS perceptual loss; 0 to disable, > 0 to scale.', type=float)

# Generator params
parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
parser.add_argument('--tile_dlatents', default=False, help='Tile dlatents to use a single vector at each scale', type=bool)
parser.add_argument('--clipping_threshold', default=1.0, help='Stochastic clipping of gradient values outside of this threshold', type=float)

# Masking params
parser.add_argument('--load_mask', default=False, help='Load segmentation masks', type=bool)
parser.add_argument('--face_mask', default=False, help='Generate a mask for predicting only the face area', type=bool)
parser.add_argument('--use_grabcut', default=True, help='Use grabcut algorithm on the face mask to better segment the foreground', type=bool)
parser.add_argument('--scale_mask', default=1.5, help='Look over a wider section of foreground for grabcut', type=float)

args = parser.parse_args()

# manual parameters
result_subdir = misc.create_result_subdir('results', 'inference_test')
misc.init_output_logging()

args.aligned_dir=os.path.join(result_subdir, args.aligned_dir)
args.dlatent_dir=os.path.join(result_subdir, args.dlatent_dir)
args.dlabel_dir=os.path.join(result_subdir, args.dlabel_dir)
args.generated_images_dir=os.path.join(result_subdir, args.generated_images_dir)

if os.path.exists(args.aligned_dir) == False:
    os.mkdir(args.aligned_dir)

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

landmarks_detector = LandmarksDetector(args.landmarks_model_path)
aligned_face_path=None
ALIGNED_IMAGES_DIR = args.aligned_dir
try:
    raw_img_path = args.image_path
    print('Getting landmarks...')
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
        try:
            print('Starting face alignment...')
            face_img_name = '%s_%02d.png' % (os.path.splitext(os.path.basename(args.image_path))[0], i)
            aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
            image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=args.resolution, x_scale=args.x_scale, y_scale=args.y_scale, em_scale=args.em_scale, alpha=args.use_alpha)
            print('Wrote result %s' % aligned_face_path)
        except:
            print("Exception in face alignment!")
except:
    print("Exception in landmark detection!")

#release memory
del landmarks_detector
gc.collect()

ref_images=aligned_face_path

beautyrater_model=beautyrater.BeautyRater(args.load_vgg_beauty_rater_model)
f1_labels=beautyrater_model.predict(ref_images)

facenet_model=facenet.FaceNet(args.load_facenet_model)
f2_labels=facenet_model.singlePredict(ref_images)

constant_labels=np.hstack([f1_labels,f2_labels]).astype(np.float32)

#release memory
del beautyrater_model
del facenet_model
gc.collect()

args.decay_steps *= 0.01 * args.iterations # Calculate steps as a percent of total iterations

os.makedirs(args.data_dir, exist_ok=True)
os.makedirs(args.mask_dir, exist_ok=True)
os.makedirs(args.generated_images_dir, exist_ok=True)
os.makedirs(args.dlatent_dir, exist_ok=True)
os.makedirs(args.dlabel_dir, exist_ok=True)

# Initialize generator and perceptual model

# load network
network_pkl = misc.locate_network_pkl(args.results_dir)
print('Loading network from "%s"...' % network_pkl)
G, D, Gs = misc.load_network_pkl(args.results_dir, None)

# initiate random input
latents = misc.random_latents(1, Gs, random_state=np.random.RandomState(800))
labels = np.random.rand(1, args.labels_size)

generator = Generator(Gs, labels_size=572, batch_size=1, clipping_threshold=args.clipping_threshold, model_res=args.resolution)

perc_model = None
if (args.use_lpips_loss > 0.00000001):
    with open(args.load_perc_model,"rb") as f:
        perc_model =  pickle.load(f)
perceptual_model = PerceptualModel(args, perc_model=perc_model, batch_size=1,constant_labels=constant_labels)
perceptual_model.build_perceptual_model(generator)

ff_model = None

# Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
name = os.path.splitext(os.path.basename(ref_images))[0]

perceptual_model.set_reference_images([ref_images])
dlatents = None
dlabels=None
if (args.load_last != ''): # load previous dlatents for initialization
    dl = np.expand_dims(np.load(os.path.join(args.load_last, f'{name}.npy')),axis=0)
    if (dlatents is None):
        dlatents = dl
    else:
        dlatents = np.vstack((dlatents,dl))
else:
    if (ff_model is None):
        if os.path.exists(args.load_resnet):
            print("Loading ResNet Model:")
            ff_model = load_model(args.load_resnet)
            from keras.applications.resnet50 import preprocess_input
    if (ff_model is None):
        if os.path.exists(args.load_effnet):
            import efficientnet
            print("Loading EfficientNet Model:")
            ff_model = load_model(args.load_effnet)
            from efficientnet import preprocess_input
    if (ff_model is not None): # predict initial dlatents with ResNet model
        dlatents = ff_model.predict(preprocess_input(load_images([ref_images],image_size=args.resnet_image_size)))

#release memory
del ff_model
gc.collect()

dlatents=np.mean(dlatents,axis=1)

# dlatents = misc.random_latents(1, Gs, random_state=np.random.RandomState(800))

if dlatents is not None:
    generator.set_dlatents(dlatents)


dlabels=np.random.rand(1, args.labels_size)

if dlabels is not None:
    generator.set_dlabels(dlabels)

op = perceptual_model.optimize([generator.dlatent_variable, generator.dlabel_variable], iterations=args.iterations)
pbar = tqdm(op, leave=False, total=args.iterations)
vid_count = 0
best_loss = None
best_dlatent = None
best_dlabel=None
history=[]

for i, loss_dict in enumerate(pbar):
    pbar.set_description(" ".join([name]) + ": " + "; ".join(["{} {:.4f}".format(k, v)
                    for k, v in loss_dict.items()]))
    if best_loss is None or loss_dict["loss"] < best_loss:
        best_loss = loss_dict["loss"]
        best_dlatent, best_dlabel= generator.get_dvariables()
    
    generator.stochastic_clip_dvariables()
    history.append((loss_dict["loss"], generator.get_dvariables()))
    
    if i % args.iterations_to_save == 0 and i > 0:
        print("saving reconstruction output for iteration num {}".format(i))
        if best_dlatent is not None and best_dlabel is not None:
            generator.get_beautify_image(dlatents=best_dlatent,dlabels=best_dlabel, index=i,dir=result_subdir)

generator.get_beautify_image(dlatents=best_dlatent,dlabels=best_dlabel, index=args.iterations,dir=result_subdir)
history.append((best_loss, best_dlatent, best_dlabel))
print(" ".join([name]), " Loss {:.4f}".format(best_loss))

# Generate images from found dlatents and save them

generator.set_dlatents(best_dlatent)
generator.set_dlabels(best_dlabel)
generated_images = generator.generate_images()
generated_dlatents,generated_dlabels = generator.get_dvariables()

for img_array, dlatent, dlabel, img_name in zip(generated_images, generated_dlatents, generated_dlabels, [name]):
    img = PIL.Image.fromarray(img_array, 'RGB')
    img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
    np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)
    np.save(os.path.join(args.dlabel_dir, f'{img_name}.npy'), dlabel)

# save history of latents
with open(result_subdir+'/history_of_latents.txt', 'w') as f:
    for item in history:
        f.write("{}\n".format(item))
        f.write("\n")

generator.reset_dlatents()
generator.reset_dlabels()
