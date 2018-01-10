#! /usr/bin/env python

import tensorflow as tf

class mnistCNN(object):
    """
    A NN for mnist classification.
    """
    def __init__(self, dense=500):
    
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, 784], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 10], name="input_y")
    
        # First layer
        self.dense_1 = self.dense_layer(self.input_x, input_dim=784, output_dim=dense)

        # Final layer
        self.dense_2 = self.dense_layer(self.dense_1, input_dim=dense, output_dim=10)

        self.predictions = tf.argmax(self.dense_2, 1, name="predictions")
        
        # Loss function
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.dense_2, labels=self.input_y))
        
        # Accuracy
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    

    def dense_layer(self, x, input_dim=10, output_dim=10, name='dense'):
        '''
        Dense layer function
        Inputs:
          x: Input tensor
          input_dim: Dimmension of the input tensor.
          output_dim: dimmension of the output tensor
          name: Layer name
        '''
        W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1), name='W_'+name)
        b = tf.Variable(tf.constant(0.1, shape=[output_dim]), name='b_'+name)
        dense_output = tf.nn.relu(tf.matmul(x, W) + b)
        return dense_output
