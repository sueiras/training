#! /usr/bin/env python

import tensorflow as tf

# Access to the data
def get_data(data_dir='/tmp/MNIST_data'):
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets(data_dir, one_hot=True)


#Batch generator
def batch_generator(mnist, batch_size=256, type='train'):
    if type=='train':
        return mnist.train.next_batch(batch_size)
    else:
        return mnist.test.next_batch(batch_size)
        