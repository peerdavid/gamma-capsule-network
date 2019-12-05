

import tensorflow as tf
from capsule.gamma_capsule_layer import GammaCapsule
from capsule.capsule_layer import Capsule
from capsule.primary_capsule_layer import PrimaryCapsule
from capsule.reconstruction_network import ReconstructionNetwork
from capsule.norm_layer import Norm
from capsule.gamma_metrics import t_score, d_score

layers = tf.keras.layers
models = tf.keras.models



class GammaCapsuleNetwork(tf.keras.Model):

    def __init__(self, n_classes=10):
        super(GammaCapsuleNetwork, self).__init__()
        self.reshape = layers.Reshape(target_shape=[28, 28, 1], input_shape=(28, 28,))
        self.conv_1 = layers.Conv2D(256, (9, 9), padding='valid', activation=tf.nn.relu)
        self.primary_1 = PrimaryCapsule()
        self.gamma_1 = GammaCapsule(in_capsules=1152, in_dim=8, out_capsules=32, out_dim=8)
        self.gamma_2 = GammaCapsule(in_capsules=32, in_dim=8, out_capsules=10, out_dim=16)
        self.reconstruction_network = ReconstructionNetwork(in_capsules=10, in_dim=16)
        self.norm = Norm()


    def call(self, x, y):
        x = self.reshape(x)
        x = self.conv_1(x)
        x = self.primary_1(x)
        v_1, c_1 = self.gamma_1(x)
        v_2, c_2 = self.gamma_2(v_1)
        
        # Network outputs
        r = self.reconstruction_network(v_2, y)
        out = self.norm(v_2)
        
        # Calculate metrics
        T = (t_score(c_1) + t_score(c_2)) / 2.0
        D = d_score(v_1)    # Note: The output activations are forced by the loss function messing the results. 
                            #       Therefore only V_1 is evaluated
        return out, r, [v_1, v_2], T, D