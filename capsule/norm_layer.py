

import tensorflow as tf
from capsule.gamma_capsule_layer import GammaCapsule
from capsule.primary_capsule_layer import PrimaryCapsule

layers = tf.keras.layers
models = tf.keras.models



class Norm(tf.keras.Model):
    def call(self, inputs):
        x = tf.norm(inputs, axis=-1)
        return x