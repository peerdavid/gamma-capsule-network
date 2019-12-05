

import tensorflow as tf
from capsule.utils import squash

layers = tf.keras.layers
models = tf.keras.models


class Capsule(tf.keras.Model):

    def __init__(self, in_capsules, in_dim, out_capsules, out_dim, routing_iterations=3, name=''):
        super(Capsule, self).__init__(name=name)
        self.in_capsules = in_capsules
        self.in_dim = in_dim
        self.out_capsules = out_capsules
        self.out_dim = out_dim
        self.routing_iterations = routing_iterations

        w_init = tf.random_normal_initializer(stddev=0.1)
        self.W = tf.Variable(initial_value=w_init(shape=(1, out_capsules, in_capsules, out_dim, in_dim),
                                                dtype='float32'),
                            trainable=True)
        
        bias_init = tf.constant_initializer(0.1)
        self.bias = tf.Variable(initial_value=bias_init(shape=(1, out_capsules, out_dim),
                                                dtype='float32'),
                            trainable=True)
    
    
    def call(self, u):
        batch_size = tf.shape(u)[0]

        # Reshape u into (batch_size, out_capsules, in_capsules, out_dim, in_dim)
        u = tf.expand_dims(u, 1) 
        u = tf.expand_dims(u, 3)  
        u = tf.tile(u, [1, self.out_capsules, 1, 1, 1])
        u = tf.tile(u, [1, 1, 1, self.out_dim, 1])

        # Duplicate transformation matrix for each batch
        w = tf.tile(self.W, [batch_size, 1, 1, 1, 1])

        # Dotwise product between u and w to get all votes
        u_hat = tf.reduce_sum(u * w, axis=-1)

        # Routing-by-agreement
        bias = tf.tile(self.bias, [batch_size, 1, 1])
        b_ij = tf.zeros(shape=[batch_size, self.out_capsules, self.in_capsules, 1])
        for r in range(self.routing_iterations):
            c_ij = tf.nn.softmax(b_ij, axis=1)
            c_ij_tiled = tf.tile(c_ij, [1, 1, 1, self.out_dim])
            s_j = tf.reduce_sum(c_ij_tiled * u_hat, axis=2) + bias
            v_j = squash(s_j)

            if(r < self.routing_iterations - 1):
                v_j = tf.expand_dims(v_j, 2)
                v_j = tf.tile(v_j, [1, 1, self.in_capsules, 1])
                u_x_v = tf.reduce_sum(v_j * u_hat, axis=-1)
                b_ij = b_ij + tf.expand_dims(u_x_v, axis=-1)

        return v_j, c_ij