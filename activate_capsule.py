###################################################################
# This script activates a single gamma-capsule to show the cause  #
# of an active capsule - https://arxiv.org/abs/1812.09707         #
#                                                                 #
# Peer David (2019)                                               #
###################################################################
try:
    import cluster_setup
except ImportError:
    pass

import os
import sys
import time
import math
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
import sklearn.metrics

from capsule.gamma_capsule_network import GammaCapsuleNetwork
import utils

#
# Hyperparameters and cmd args
#
argparser = argparse.ArgumentParser(description="Generate images that activate a single gamma-capsule") 
argparser.add_argument("--log_dir", default="experiments/robust", 
  help="Learning rate of adam")           
argparser.add_argument("--num_avg", default=5, type=int, 
  help="How many generated images should be used to calc the average")
argparser.add_argument("--num_samples", default=60, type=int, 
  help="How many images should be generated")
argparser.add_argument("--ckpt", default=10, type=int, 
  help="Checkpoint to use (epoch)")
argparser.add_argument("--layer", default=1, type=int, 
  help="How many generated images should be used to calc the average")
argparser.add_argument("--num_classes", default=10, type=int, 
  help="Number of classes for network")

# Load hyperparameters from cmd args and update with json file
args = argparser.parse_args()

#
# Functions
#
def get_img_for_caps(x, model, caps_of_interest):
  @tf.function
  def step(x, model):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(x)

      logits, _, layers, _, _ = model(x, [0]*args.num_samples)
      layer = layers[args.layer]
      capsule_layer = tf.norm(layer, axis=-1)
      num_capsules_of_layer = tf.shape(layer)[1]
      active_capsule_one_hot = tf.one_hot(caps_of_interest, depth=num_capsules_of_layer)
      
      loss = tf.reduce_sum(tf.square((active_capsule_one_hot * capsule_layer) - active_capsule_one_hot), axis=-1)   
      loss = loss + 1e-5 * (tf.reduce_sum(x, axis=[1, 2, 3]))

    dl_dx = tape.gradient(loss, x)
    x -= 0.01 * tf.sign(dl_dx)
    x = tf.clip_by_value(x, 0.0, 1.0)
    
    return x, loss, logits, num_capsules_of_layer

  for _ in range(1000):
    x, loss, logits, num_capsules_of_layer = step(x, model)

  # Keep only the 5 best examples and show avg
  sorted_loss = tf.sort(loss, axis=-1, direction='ASCENDING')
  smallest = sorted_loss[args.num_avg-1]
  keep = tf.cast(tf.math.less_equal(loss, smallest), tf.float32)
 
  # Calculate avg logits for the n best images
  logits_to_keep = tf.expand_dims(keep, -1)
  logits_to_keep = tf.tile(logits_to_keep, [1, args.num_classes])
  logits *= logits_to_keep
  logits = tf.reduce_sum(logits, axis=0)
  logits = logits / float(args.num_avg)

  # Calculate avg image out of n best images
  images_to_keep = tf.expand_dims(keep, -1)
  images_to_keep = tf.expand_dims(images_to_keep, -1)
  images_to_keep = tf.tile(images_to_keep, [1, 28, 28])
  images_to_keep = tf.expand_dims(images_to_keep, -1)
  x *= images_to_keep
  x = tf.reduce_sum(x, axis=0)
  x = x / float(args.num_avg)

  return x, logits, num_capsules_of_layer


#
# M A I N
#
def main():
  # Load model from checkpoint
  writer = tf.summary.create_file_writer("%s/log/generated" % args.log_dir)
  optimizer = optimizers.Adam()
  model = GammaCapsuleNetwork(args.num_classes)
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  checkpoint.restore("%s/ckpt/ckpt-%d" % (args.log_dir, args.ckpt))
  
  # Generate images that activate a capsule
  # Note: The batch size defines how many samples we collect i.e. 60
  caps = 0
  while True:
    x = tf.random.uniform((args.num_samples, 28, 28, 1), minval=0.1, maxval=0.9)

    with tf.device("/GPU:1"):
      print("Layer %d | Capsule %d" % (args.layer, caps))
      x, logits, num_capsules = get_img_for_caps(x, model, caps)
    
    with writer.as_default(): 
      x = tf.squeeze(x)
      img = utils.plot_generated_image(x, logits)
      x = utils.plot_to_image(img)
      tf.summary.image("Layer %d/Capsule %d" % (args.layer, caps), x, step=args.ckpt)
    
    caps += 1
    if caps >= num_capsules:
      break


if __name__ == '__main__':
    main()