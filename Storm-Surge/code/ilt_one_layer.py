from __future__ import print_function
import numpy as np
import os
import sys
import time
import tensorflow as tf
import datetime as dt
import ilt_aux_model as aux

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('hidden1', 80, 'Size of the first hidden layer')
flags.DEFINE_integer('output_vars', 10, 'Size of the output layer')
flags.DEFINE_integer('input_vars', 6, 'Size of the input layer')

# Learning rate is important for model training. 
# Decrease learning rate for more complicated models.
# Increase if convergence is steady but too slow
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
flags.DEFINE_float('learning_rate_decay', 0.5, 'Learning rate decay, i.e. the fraction of the initial learning rate at the end of training')
flags.DEFINE_integer('max_steps', 10001, 'Number of steps to run trainer')
flags.DEFINE_float('max_loss', 0.01, 'Max acceptable validation MSE')
flags.DEFINE_float('moving_avg_decay', 0.999, 'Moving average decay for training variables')

def inference(inputs):
    """
    Build the graph/ANN to make predictions.
    inputs -- tensor representing network inputs
    train_prediction (return) -- network outputs
    """
    hidden_1 = aux.nn_layer(inputs, FLAGS.input_vars, FLAGS.hidden1, 'layer1')
    train_prediction = aux.nn_layer(hidden_1, FLAGS.hidden1, FLAGS.output_vars, 'output', act = tf.nn.elu)      
    return train_prediction

def loss(nn_outputs, true_outputs):
    """
    Adds loss function to the graph/ANN
    nn_outputs -- tensor representing network outputs
    true_outputs -- tensor representing true (training) outputs
    MSE -- Mean Squared Error (MSE), i.e. the losses tensor
    """
    prediction_diff = nn_outputs-true_outputs
    MSE = tf.cast(tf.reduce_mean(tf.reduce_mean(tf.square(prediction_diff))),tf.float32)

    # Save MSE to the collection 
    tf.add_to_collection('losses',MSE)
    return MSE
    
def training(MSE, learning_rate):
    """
    Adds to the loss model the Ops required to generate and apply gradients
    MSE -- Mean Squared Error (MSE), i.e. the losses tensor
    learning_rate -- learning rate for training
    """

    # create optimizer. Other options are GradientDescentOptimizer, AdagradOptimizer, etc.
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # calculate gradients
    gradients = optimizer.compute_gradients(MSE)

    # create/reuse a variable on CPU to keep track of number of iterations at training
    #with tf.device('/cpu:0'):
    #    initial = tf.constant_initializer(0)
    #    global_step = tf.get_variable('global_step',shape = 0, initializer = initial, trainable = False)
    global_step = tf.Variable(0.00, trainable=False)

    # Add tensor to apply calculated gradients. 
    # Use sess.run([apply_gradient_op]) to perform one training iteration
    apply_gradient_op = optimizer.apply_gradients(gradients, global_step = global_step)
    return apply_gradient_op

def evaluation(nn_outputs, true_outputs):
    """
    Added for compatibility only
    """
    return loss(nn_outputs, true_outputs)
