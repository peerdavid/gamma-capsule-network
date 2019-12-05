import tensorflow as tf
import numpy as np


def t_score(c_ij):
    """ Calculates the T-score to measure whether capsules are 
        coupled in a tree structure (~1) or not (~0) [1]

        param: c_ij - coupling coefficient with shape (batch_size, out_caps, in_caps, 1) 
    """
    out_caps = tf.cast(tf.shape(c_ij)[1], tf.float32)
    c_ij = tf.squeeze(c_ij, axis=3)         # (batch_size, out_caps, in_caps) 
    c_ij = tf.transpose(c_ij, [0, 2, 1])    # (batch_size, in_caps, out_caps) 

    epsilon = 1e-12
    entropy = -tf.reduce_sum(c_ij * tf.math.log(c_ij + epsilon), axis=-1)
    T = 1 - entropy / -tf.math.log(1 / out_caps)
    return tf.reduce_mean(T)


def d_score(v_j):
    """ Measures how the activation of capsules adapts to the input.

        param: v_j - activations of capsules with shape (batch_size, num_capsules, dim)
    """
    v_j_norm = tf.norm(v_j, axis=-1)
    v_j_std = tf.math.reduce_std(v_j_norm, axis=0)   # Note: Calc std along the batch dimension
    return tf.reduce_max(v_j_std)    


def v_map(v_j):
    """ Creates an activation map for the given activations

        param: v_j - activations of capsules with shape (batch_size, num_capsules, dim)
    """
    v_j_norm = tf.norm(v_j, axis=-1)
    v_j_norm = tf.expand_dims(v_j_norm, 0)      # "batch" of one image map
    v_j_norm = tf.expand_dims(v_j_norm, -1)     # "RGB"
    return v_j_norm