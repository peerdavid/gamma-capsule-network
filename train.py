###################################################################
# Original training implementation of gamma-capsules              #
# using TensorFlow 2.0 - https://arxiv.org/abs/1812.09707         #
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

import utils
import attack
from capsule.gamma_capsule_network import GammaCapsuleNetwork
from capsule.utils import margin_loss
from capsule.gamma_metrics import v_map


#
# Hyperparameters and cmd args
#
argparser = argparse.ArgumentParser(description="Train gamma-capsule networks on multiple GPUs")
argparser.add_argument("--learning_rate", default=0.001, type=float, 
  help="Learning rate of adam")
argparser.add_argument("--reconstruction_weight", default=0.0005, type=float, 
  help="Learning rate of adam")
argparser.add_argument("--log_dir", default="experiments/robust", 
  help="Learning rate of adam")    
argparser.add_argument("--batch_size", default=32, type=int, 
  help="Learning rate of adam")
argparser.add_argument("--enable_tf_function", default=True, type=bool, 
  help="Enable tf.function for faster execution")
argparser.add_argument("--num_classes", default=10, type=int, 
  help="Number of classes of the training set")
argparser.add_argument("--epochs", default=10, type=int, 
  help="Defines the number of epochs to train the network")
argparser.add_argument("--gamma_robust", default=True, type=bool, 
  help="Training to learn gamma-robust useful features")

# Load hyperparameters from cmd args and update with json file
args = argparser.parse_args()


#
# Functions
#
def create_fashion_mnist():
  """ Create pipeline for fashionMNIST without random data-augmentation as written in [1]
  """

  # Import fashionMNIST dataset
  fashion_mnist = keras.datasets.fashion_mnist
  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

  # No random data augmentation is used [1]
  train_images = train_images.astype(np.float32)
  train_images = train_images / 255.0
  train_labels = train_labels.astype(np.int64)
  train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
  train_ds = train_ds.shuffle(60000).batch(args.batch_size)

  test_images = test_images.astype(np.float32)
  test_images = test_images / 255.0
  test_labels = test_labels.astype(np.int64)
  test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
  
  # Create [0] test dataset
  def filter_fn(x, y):
      return tf.math.equal(y, 0) 
  test_ds_0 = test_ds.filter(filter_fn)  

  test_ds = test_ds.batch(100)
  test_ds_0 = test_ds_0.batch(100)

  class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  
  return train_ds, (test_ds, test_ds_0), class_names


def compute_loss(logits, y, reconstruction, x):
  """ The loss is the sum of the margin loss and the reconstruction loss 
      as defined in [2]
  """ 
  num_classes = tf.shape(logits)[1]

  # Calculate margin loss
  loss = margin_loss(logits, tf.one_hot(y, num_classes))
  loss = tf.reduce_mean(loss)

  # Calculate reconstruction loss
  x_1d = tf.keras.layers.Flatten()(x)
  distance = tf.square(reconstruction - x_1d)
  reconstruction_loss = tf.reduce_sum(distance, axis=-1)
  reconstruction_loss = args.reconstruction_weight * tf.reduce_mean(reconstruction_loss)

  loss = loss + reconstruction_loss

  return loss, reconstruction_loss


def train(train_ds, all_test_ds, class_names):
  """ Train gamma-capsule networks mirrored on multiple gpu's
  """
  test_ds, test_ds_0 = all_test_ds

  # Run training for multiple epochs mirrored on multiple gpus
  strategy = tf.distribute.MirroredStrategy()
  num_replicas = strategy.num_replicas_in_sync
  train_ds = strategy.experimental_distribute_dataset(train_ds)
  test_ds = strategy.experimental_distribute_dataset(test_ds)
  test_ds_0 = strategy.experimental_distribute_dataset(test_ds_0)

  # Create a checkpoint directory to store the checkpoints.
  ckpt_dir = os.path.join(args.log_dir, "ckpt/", "ckpt")

  train_writer = tf.summary.create_file_writer("%s/log/train" % args.log_dir)
  test_writer = tf.summary.create_file_writer("%s/log/test [0-9]" % args.log_dir)
  test_writer_0 = tf.summary.create_file_writer("%s/log/test [0]" % args.log_dir)

  with strategy.scope():
    model = GammaCapsuleNetwork(args.num_classes)
    optimizer = optimizers.Adam(learning_rate=args.learning_rate)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    # Define metrics 
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    train_t_score = tf.keras.metrics.Mean(name='train_t_score')
    train_d_score = tf.keras.metrics.Mean(name='train_d_score')
    
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_t_score = tf.keras.metrics.Mean(name='test_t_score')
    test_d_score = tf.keras.metrics.Mean(name='test_d_score')
    
    # Function for a single training step
    def train_step(inputs):
      # Note: Here we do emperical risk minimization under attack
      x, y = inputs
      x_adv = attack.pgd(x, y, model, eps=0.1, a=0.01, k=40) if args.gamma_robust else x      
      with tf.GradientTape() as tape:
        logits, reconstruction, _, T, D = model(x_adv, y)

        # We want to reconstruct the original x rather than x_adv , because 
        # small pert. should represent the same image
        loss, _ = compute_loss(logits, y, reconstruction, x)
      
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      train_accuracy.update_state(y, logits)
      train_t_score.update_state(T)
      train_d_score.update_state(D)
      return loss, (x, x_adv, reconstruction)

    # Function for a single test step
    def test_step(inputs):
      x, y = inputs
      logits, reconstruction, layers, T, D = model(x, y)
      loss, _ = compute_loss(logits, y, reconstruction, x)
      
      test_accuracy.update_state(y, logits)
      test_loss.update_state(loss)
      test_t_score.update_state(T)
      test_d_score.update_state(D)

      pred = tf.math.argmax(logits, axis=1)
      cm = tf.math.confusion_matrix(y, pred, num_classes=args.num_classes)

      return cm, layers

    # Define functions for distributed training
    def distributed_train_step(dataset_inputs):
      return strategy.experimental_run_v2(train_step,
                                                        args=(dataset_inputs,))
      #return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

    def distributed_test_step(dataset_inputs):
      return strategy.experimental_run_v2(test_step, args=(dataset_inputs, ))
    
    if args.enable_tf_function:
      distributed_train_step = tf.function(distributed_train_step)
      distributed_test_step = tf.function(distributed_test_step)

    # Loop for multiple epochs
    step = 0
    for epoch in range(args.epochs):
      ########################################
      # Test [0-9]
      ########################################
      cm = np.zeros((args.num_classes, args.num_classes))
      for data in test_ds:
        distr_cm, distr_layers = distributed_test_step(data)
        for r in range(num_replicas):
          cm += distr_cm.values[r]

      # Log test results (for replica 0 only for activation map and reconstruction)
      figure = utils.plot_confusion_matrix(cm.numpy(), class_names)
      cm_image = utils.plot_to_image(figure)
      with test_writer.as_default(): 
        tf.summary.image("Confusion Matrix", cm_image, step=step)

      with test_writer.as_default(): 
        tf.summary.image(
          "Activation Map",
          v_map(distr_layers[0].values[r]),
          step=step,
          max_outputs=1,)

      print("TEST [0-9] | epoch %d (%d): acc=%.2f, loss=%.3f, T=%.2f, D=%.2f" % 
            (epoch, step, test_accuracy.result(), test_loss.result(), test_t_score.result(), 
            test_d_score.result()), flush=True)  

      with test_writer.as_default(): 
        tf.summary.scalar("General/Accuracy", test_accuracy.result(), step=step)
        tf.summary.scalar("General/Loss", test_loss.result(), step=step)
        tf.summary.scalar("Gamma-Metrics/T-Score", test_t_score.result(), step=step)
        tf.summary.scalar("Gamma-Metrics/D-Score", test_d_score.result(), step=step)
      test_accuracy.reset_states()
      test_loss.reset_states()
      test_t_score.reset_states()
      test_d_score.reset_states()
      test_writer.flush()

      ########################################
      # Test [0]
      ########################################
      for data in test_ds_0:
        _, distr_layers = distributed_test_step(data)

      # Log test results
      with test_writer_0.as_default(): 
        tf.summary.image(
          "Activation Map",
          v_map(distr_layers[0].values[0]),
          step=step,
          max_outputs=1,)

      print("TEST [0] | epoch %d (%d): acc=%.2f, loss=%.3f, T=%.2f, D=%.2f" % 
            (epoch, step, test_accuracy.result(), test_loss.result(), test_t_score.result(), 
            test_d_score.result()), flush=True)  
      with test_writer_0.as_default(): 
        tf.summary.scalar("General/Accuracy", test_accuracy.result(), step=step)
        tf.summary.scalar("General/Loss", test_loss.result(), step=step)
        tf.summary.scalar("Gamma-Metrics/T-Score", test_t_score.result(), step=step)
        tf.summary.scalar("Gamma-Metrics/D-Score", test_d_score.result(), step=step)

      test_accuracy.reset_states()
      test_loss.reset_states()
      test_t_score.reset_states()
      test_d_score.reset_states()
      test_writer.flush()

      ########################################
      # Train
      ########################################
      for data in train_ds:
        start = time.time()
        distr_loss, distr_imgs = distributed_train_step(data)
        train_loss = 0
        for r in range(num_replicas):
          train_loss += distr_loss.values[r]        

        if step % 100 == 0:
          # Show some inputs, adversarial inputs and reconstructions
          time_per_step = (time.time()-start) * 1000 / 100
          print("TRAIN | epoch %d (%d): acc=%.2f, loss=%.3f, T=%.2f, D=%.2f | Time per step[ms]: %.2f" % 
              (epoch, step, train_accuracy.result(), train_loss.numpy(), 
                train_t_score.result(), train_d_score.result(), time_per_step), flush=True)     

          # Create recon tensorboard images
          x, x_adv, recon_x = distr_imgs[0].values[0], distr_imgs[1].values[0], distr_imgs[2].values[0]
          recon_x = tf.reshape(recon_x, [-1, tf.shape(x)[1], tf.shape(x)[2]])  
          img = tf.concat([x, x_adv, recon_x], axis=1)
          img = tf.expand_dims(img, -1)

          with train_writer.as_default(): 
            tf.summary.scalar("General/Accuracy", train_accuracy.result(), step=step)
            tf.summary.scalar("General/Loss", train_loss.numpy(), step=step)
            tf.summary.scalar("Gamma-Metrics/T-Score", train_t_score.result(), step=step)
            tf.summary.scalar("Gamma-Metrics/D-Score", train_d_score.result(), step=step)
            tf.summary.image(
              "X & XAdv & Recon",
              img,
              step=step,
              max_outputs=3,)

          train_accuracy.reset_states()
          train_t_score.reset_states()
          train_d_score.reset_states()
          start = time.time()

          train_writer.flush()

        step += 1
      
      ####################
      # Checkpointing
      if epoch % 1 == 0:
        checkpoint.save(ckpt_dir)


#
# M A I N
#
def main():
  # Configurations for cluster
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  for r in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[r], True)

  # Write log folder and arguments
  if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

  with open("%s/args.txt" % args.log_dir, "w") as file:
     file.write(json.dumps(vars(args)))

  # Load data
  train_ds, test_ds, class_names = create_fashion_mnist()

  # Train gamma-capsule network
  train(train_ds, test_ds, class_names)


       
if __name__ == '__main__':
    main()