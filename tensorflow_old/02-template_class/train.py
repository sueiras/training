#! /usr/bin/env python

from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import tensorflow as tf

from data_utils import get_data, batch_generator
from model_mnist_cnn import mnistCNN


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("data_directory", '/tmp/MNIST_data', "Data dir (default /tmp/MNIST_data)")

# Model Hyperparameters
tf.flags.DEFINE_integer("dense_size", 500, "dense_size (default 500)")

# Training parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate (default: 0.001)")
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 256)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 20)")

# Misc Parameters
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

#Access to the data
mnist_data = get_data(data_dir= FLAGS.data_directory)


# Training
# ==================================================

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333, allow_growth = True)
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        
        # Create model
        cnn = mnistCNN(dense=FLAGS.dense_size)
        
        # Trainer
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cnn.loss)

        # Saver
        saver = tf.train.Saver(max_to_keep=1)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Train proccess
        for epoch in range(FLAGS.num_epochs):
            for n_batch in range(int(55000/FLAGS.batch_size)):
                batch = batch_generator(mnist_data, batch_size=FLAGS.batch_size, type='train')
                _, ce = sess.run([train_op, cnn.loss], feed_dict={cnn.input_x: batch[0], cnn.input_y: batch[1]})

            print(epoch, ce)
        model_file = saver.save(sess, '/tmp/mnist_model')
        print('Model saved in', model_file)

