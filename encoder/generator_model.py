import math
import tensorflow as tf
import numpy as np
import dnnlib.tflib as tflib
from functools import partial
import misc
import PIL
from PIL import Image
import os 

class Generator:
    def __init__(self, model, labels_size=572, batch_size=1, clipping_threshold=1, model_res=128):
        self.batch_size = batch_size
        self.clipping_threshold=clipping_threshold
        self.initial_dlatents = misc.random_latents(1, model, random_state=np.random.RandomState(800)) #np.zeros((self.batch_size, 512))
        self.initial_dlabels = np.random.rand(self.batch_size, labels_size)
        self.sess = tf.get_default_session()
        self.graph = tf.get_default_graph()

        def get_tensor(name):
            try:
                return self.graph.get_tensor_by_name(name)
            except KeyError:
                return None

        self.dlatent_variable = tf.get_variable('learnable_dlatents',
            shape=(batch_size, 512),
            dtype='float32',
            initializer=tf.initializers.random_normal())

        self.dlabel_variable = tf.get_variable('learnable_dlabels',
            shape=(batch_size, labels_size),
            dtype='float32',
            initializer=tf.initializers.random_normal())
        
        self.generator_output = model.get_output_for(self.dlatent_variable, self.dlabel_variable)

        self.latents_name_tensor = get_tensor(model.input_templates[0].name)
        self.labels_name_tensor = get_tensor(model.input_templates[1].name)
        self.output_name_tensor = get_tensor(model.output_templates[0].name)

        self.output_name_image= tflib.convert_images_to_uint8(self.output_name_tensor, nchw_to_nhwc=True, uint8_cast=False)
        self.output_name_image_uint8 = tf.saturate_cast(self.output_name_image, tf.uint8)
        

        self.set_dlatents(self.initial_dlatents)
        self.set_dlabels(self.initial_dlabels)

        self.generator_output_shape=model.output_shape

        if self.generator_output is None:
            for op in self.graph.get_operations():
                print(op)
            raise Exception("Couldn't find generator_output")

        self.generated_image = tflib.convert_images_to_uint8(self.generator_output, nchw_to_nhwc=True, uint8_cast=False)
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8)

        # Implement stochastic clipping similar to what is described in https://arxiv.org/abs/1702.04782
        # (Slightly different in that the latent space is normal gaussian here and was uniform in [-1, 1] in that paper,
        # so we clip any vector components outside of [-2, 2]. It seems fine, but I haven't done an ablation check.)
        clipping_mask1 = tf.math.logical_or(self.dlatent_variable > self.clipping_threshold, self.dlatent_variable < -self.clipping_threshold)
        clipped_values1 = tf.where(clipping_mask1, tf.random_normal(shape=(self.batch_size, 512)), self.dlatent_variable)
        self.stochastic_clip_op1 = tf.assign(self.dlatent_variable, clipped_values1)

        clipping_mask2_1 = tf.math.logical_or(self.dlabel_variable[:,0:60] > self.clipping_threshold, self.dlabel_variable[:,0:60] < 0)
        clipping_mask2_2 = tf.math.logical_or(self.dlabel_variable[:,60:] > self.clipping_threshold, self.dlabel_variable[:,60:] < -self.clipping_threshold)
        clipping_mask2 = tf.concat([clipping_mask2_1,clipping_mask2_2],axis=1)
        clipped_values2 = tf.where(clipping_mask2, tf.random_normal(shape=(self.batch_size, labels_size)), self.dlabel_variable)
        self.stochastic_clip_op2 = tf.assign(self.dlabel_variable, clipped_values2)


    def reset_dlatents(self):
        self.set_dlatents(self.initial_dlatents)
    
    def reset_dlabels(self):
        self.set_dlabels(self.initial_dlabels)

    def set_dlatents(self, dlatents):
        self.sess.run(tf.assign(self.dlatent_variable, dlatents))
    
    def set_dlabels(self, dlabels):
        self.sess.run(tf.assign(self.dlabel_variable, dlabels))

    def stochastic_clip_dvariables(self):
        self.sess.run([self.stochastic_clip_op1, self.stochastic_clip_op2])

    def get_dvariables(self):
        return self.sess.run([self.dlatent_variable, self.dlabel_variable])

    def generate_images(self, dlatents=None):
        if dlatents:
            self.set_dlatents(dlatents)
        return self.sess.run(self.generated_image_uint8)
    
    def get_beautify_image(self, dlatents=None,dlabels=None, index=None,dir=None):
        for k in range(10):
            y_pred = dlabels[:]
            y_pred[:,0:60] = y_pred[:,0:60] + (k*0.05)
            y_pred[:,0:60] = np.clip(y_pred[:,0:60], 0.0, 1.0)
            img=self.sess.run(self.output_name_image_uint8, feed_dict={self.labels_name_tensor: y_pred, self.latents_name_tensor: dlatents})
            img = PIL.Image.fromarray(img[0], 'RGB')
            img.save(os.path.join(dir, '{}_{}.png'.format(index,k)), 'PNG')
    
    def get_generate_image_shape(self):
        return self.generator_output_shape
