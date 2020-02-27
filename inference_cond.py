import os
import misc
import numpy as np
import pdb
from config import EasyDict
import tfutil
import argparse

from random import gauss

# initialize parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', '-results_dir', help='name of training experiment folder', default='dean_cond_batch16', type=str)
parser.add_argument('--outputs', '-outputs', help='how many sequences to print', default=500, type=int)
parser.add_argument('--labels_size', '-labels_size', help='size of labels vector', default=60, type=int)
parser.add_argument('--beauty_levels', '-beauty_levels', help='number of possible beauty levels', default=5, type=int)
parser.add_argument('--total_agreement', dest='total_agreement', help='all voters agreed on same beauty level',action='store_true')
parser.add_argument('--classification', dest='classification', help='if asked, use classification conditioning instead of original', action='store_true')
args = parser.parse_args()

# manual parameters
result_subdir = misc.create_result_subdir('results', 'inference_test')
misc.init_output_logging()

# initialize TensorFlow
print('Initializing TensorFlow...')
env = EasyDict() # Environment variables, set by the main program in train.py.
env.TF_CPP_MIN_LOG_LEVEL = '1' # Print warnings and errors, but disable debug info.
env.CUDA_VISIBLE_DEVICES = '0' # Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use. change to '0' if first GPU is better
os.environ.update(env)
tf_config = EasyDict() # TensorFlow session config, set by tfutil.init_tf().
tf_config['graph_options.place_pruned_graph'] = True # False (default) = Check that all ops are available on the designated device.
tf_config['gpu_options.allow_growth']          = True     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
tf_config['gpu_options.per_process_gpu_memory_fraction']          = 0.5
tfutil.init_tf(tf_config)

#load network
network_pkl = misc.locate_network_pkl(args.results_dir)
print('Loading network from "%s"...' % network_pkl)
G, D, Gs = misc.load_network_pkl(args.results_dir, None)

# sample <args.output> sequences
for j in range(args.outputs):
    
    # change the random seed in every iteration
    np.random.seed(j)
    
    # generate random noise
    latents = misc.random_latents(1, Gs, random_state=np.random.RandomState(j))
    
    # if classification asked, perform conditioning using classification vector
    if args.classification:
        for i in range(args.labels_size):
            
            # initiate conditioned label
            labels = np.zeros([1, args.labels_size], np.float32)
            labels[0][i] = 1.0

            vector = [gauss(0, 1) for j in range(512)]
            mag = sum(x ** 2 for x in vector) ** .5
            id_vectors=[x/mag for x in vector]
            id_vectors = np.expand_dims(np.array(id_vectors, dtype=np.float32),axis=0)
            combined_labels = np.hstack((labels, id_vectors)).astype(np.float32)
            
            # infer conditioned noise to receive image
            image = Gs.run(latents, combined_labels, minibatch_size=1, num_gpus=1, out_mul=127.5, out_add=127.5, out_shrink=1, out_dtype=np.uint8)
            #image = Gs.run(latents, labels, minibatch_size=1, num_gpus=0, out_mul=127.5, out_add=127.5, out_shrink=1, out_dtype=np.uint8)
            
            # save generated image as 'i.png' and noise vector as noise_vector.txt
            misc.save_image_grid(image, os.path.join(result_subdir, '{}_{}.png'.format('%04d' % j,i)), [0,255], [1,1])
            
            # save latent space for later use
            np.save(os.path.join(result_subdir,'latents_vector.npy'), latents)
            
    # classification is not asked, we will use varied conditioning vector
    else:
    
        min_beauty_level = 1.0 / args.beauty_levels
        std = min_beauty_level / 2.0 - (min_beauty_level / 10.0)
        for i in range(args.beauty_levels):
            
            # initiate beauty rates label
            if args.total_agreement:
                labels = np.ones(args.labels_size)
                labels = labels * (min_beauty_level * (i +1))
                labels = np.expand_dims(labels, axis=0)
            else:
                labels = np.random.normal(min_beauty_level*(i+1), std, [1, args.labels_size])
                labels = np.clip(labels, 0.0, 1.0)
            
            vector = [gauss(0, 1) for j in range(512)]
            mag = sum(x ** 2 for x in vector) ** .5
            id_vectors=[x/mag for x in vector]
            id_vectors = np.expand_dims(np.array(id_vectors, dtype=np.float32), axis=0)
            combined_labels = np.hstack((labels, id_vectors)).astype(np.float32)

            # infer conditioned noise to receive image
            image = Gs.run(latents, combined_labels, minibatch_size=1, num_gpus=1, out_mul=127.5, out_add=127.5, out_shrink=1, out_dtype=np.uint8)
            #image = Gs.run(latents, labels, minibatch_size=1, num_gpus=0, out_mul=127.5, out_add=127.5, out_shrink=1, out_dtype=np.uint8)
            
            # save generated image as 'i.png' and noise vector as noise_vector.txt
            misc.save_image_grid(image, os.path.join(result_subdir, '{}_{}.png'.format('%04d' % j,i)), [0,255], [1,1])
            
            # save latent space for later use
            np.save(os.path.join(result_subdir,'latents_vector.npy'), latents)
    
    if j % 10 == 0:
        print("saved {}/{} images".format(j,args.outputs))

