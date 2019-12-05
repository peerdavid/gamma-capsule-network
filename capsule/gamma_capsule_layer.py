

import tensorflow as tf
from capsule.utils import squash
import numpy as np

layers = tf.keras.layers
models = tf.keras.models


class GammaCapsule(tf.keras.Model):

    def __init__(self, in_capsules, in_dim, out_capsules, out_dim, routing_iterations=3, name=''):
        super(GammaCapsule, self).__init__(name=name)
        self.in_capsules = in_capsules
        self.in_dim = in_dim
        self.out_capsules = out_capsules
        self.out_dim = out_dim
        self.routing_iterations = routing_iterations

        w_init = tf.keras.initializers.TruncatedNormal(stddev=0.1)
        self.W = tf.Variable(initial_value=w_init(shape=(1, out_capsules, in_capsules, out_dim, in_dim),
                                                dtype='float32'),
                            trainable=True)
        
        bias_init = tf.constant_initializer(0.1)
        self.bias = tf.Variable(initial_value=bias_init(shape=(1, out_capsules, out_dim),
                                                dtype='float32'),
                            trainable=True)
    
    
    def call(self, u):
        """
            param: u - (batch_size, in_caps, in_dim)
        """
        batch_size = tf.shape(u)[0]
        u_norm = tf.norm(u, axis=-1)     # (batch_size, in_caps)

        # Reshape u into (batch_size, out_caps, in_caps, out_dim, in_dim)
        u = tf.expand_dims(u, 1) 
        u = tf.expand_dims(u, 3)  
        u = tf.tile(u, [1, self.out_capsules, 1, 1, 1])
        u = tf.tile(u, [1, 1, 1, self.out_dim, 1])

        # Duplicate transformation matrix for each batch
        w = tf.tile(self.W, [batch_size, 1, 1, 1, 1])

        # Dotwise product between u and w to get all votes
        # shape = (batch_size, out_caps, in_caps, out_dim)
        u_hat = tf.reduce_sum(u * w, axis=-1)

        # Ensure that ||u_hat|| <= ||v_i||
        u_hat_norm = tf.norm(u_hat, axis=-1, keepdims=True)
        u_norm = tf.expand_dims(u_norm, axis=1)
        u_norm = tf.expand_dims(u_norm, axis=3)
        u_norm = tf.tile(u_norm, [1, self.out_capsules, 1, self.out_dim])
        new_u_hat_norm = tf.math.minimum(u_hat_norm, u_norm)
        u_hat = u_hat / u_hat_norm * new_u_hat_norm

        # Scaled-distance-agreement routing
        bias = tf.tile(self.bias, [batch_size, 1, 1])
        b_ij = tf.zeros(shape=[batch_size, self.out_capsules, self.in_capsules, 1])
        for r in range(self.routing_iterations):
            c_ij = tf.nn.softmax(b_ij, axis=1)
            c_ij_tiled = tf.tile(c_ij, [1, 1, 1, self.out_dim])
            s_j = tf.reduce_sum(c_ij_tiled * u_hat, axis=2) + bias
            v_j = squash(s_j)

            if(r < self.routing_iterations - 1):
                v_j = tf.expand_dims(v_j, 2)
                v_j = tf.tile(v_j, [1, 1, self.in_capsules, 1]) # (batch_size, out_caps, in_caps, out_dim)
                
                # Calculate scale factor t
                p_p = 0.9
                d = tf.norm(v_j - u_hat, axis=-1, keepdims=True)
                d_o = tf.reduce_mean(tf.reduce_mean(d))
                d_p = d_o * 0.5
                t = tf.constant(np.log(p_p * (self.out_capsules - 1)) - np.log(1 - p_p), dtype=tf.float32) \
                     / (d_p - d_o + 1e-12)
                t = tf.expand_dims(t, axis=-1)

                # Calc log prior using inverse distances
                b_ij = t * d

        return v_j, c_ij