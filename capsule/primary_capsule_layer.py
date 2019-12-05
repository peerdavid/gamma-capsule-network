

import tensorflow as tf
from capsule.utils import squash

layers = tf.keras.layers
models = tf.keras.models


class PrimaryCapsule(tf.keras.Model):

    def __init__(self, channels=32, dim=8, kernel_size=(9, 9), strides=2, name=''):
        """ "The second layer (PrimaryCapsules) is a convolutional capsule layer with 32 channels of convolutional
             8D capsules (i.e. each primary capsule contains 8 convolutional units with a 9 × 9 kernel and a stride
             of 2)" (Sabour et al 2017)
        """
        super(PrimaryCapsule, self).__init__(name=name)
        assert channels % dim == 0, "Invalid size of channels and dim_capsule"

        num_filters = channels * dim
        self.conv1 = layers.Conv2D(
            filters=num_filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            activation=None,    # Squashing is used as nonlinearity
            padding='valid')
        
        self.reshape = layers.Reshape(target_shape = (-1, dim))


    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.reshape(x)     # Shape=(batch_size, 1152, 8)
        x = squash(x)
        return x