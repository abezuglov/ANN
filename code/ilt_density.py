from __future__ import print_function
import numpy as np
import os
import sys
import time
import math
import tensorflow as tf
import datetime as dt
import ilt_aux_model as aux

flags = tf.app.flags
FLAGS = flags.FLAGS

# Structure of the network
flags.DEFINE_integer('hidden', 80, 'Size of the first hidden layer')
flags.DEFINE_integer('output_vars', 10, 'Size of the output layer') # no. of output vars
flags.DEFINE_integer('input_vars', 6, 'Size of the input layer')

# Learning rate is important for model training. 
# Decrease learning rate for more complicated models.
# Increase if convergence is steady but too slow
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_float('learning_rate_decay', 0.9, 'Learning rate decay, i.e. the fraction of the initial learning rate at the end of training')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer')
flags.DEFINE_float('moving_avg_decay', 0.999, 'Moving average decay for training variables')


def inference(inputs):
    """
    Build the graph/ANN to make predictions.
    inputs -- tensor representing network inputs
    train_prediction (return) -- network outputs, i.e. mean_0, mean_1, ..., mean_n, sigma_0, sigma_1, ..., sigma_n
    """
    hidden = aux.nn_layer(inputs, FLAGS.input_vars, FLAGS.hidden, 'layer1')

    train_prediction = aux.nn_layer(hidden, FLAGS.hidden, 
                                    FLAGS.output_vars*2, # Return mu's and sigma's for each output
                                    'output')#, act = None) 

    # split into two parts by the second (1) dimension
    out_mu, out_sigma = tf.split(1,2,train_prediction) 
    out_sigma = tf.exp(out_sigma) # make sure that sigma is not less than 0
    train_prediction = tf.concat(1, [out_mu, out_sigma]) # concatenate the outputs

    return train_prediction

def loss(nn_outputs, true_outputs):
    """
    Adds loss function to the graph/ANN
    nn_outputs -- tensor representing network outputs. Note its size: (mu,sigma)*FLAGS.output_vars
    true_outputs -- tensor representing true (training) outputs. Size FLAGS.output_vars
    MSE -- Mean Squared Error (MSE), i.e. the losses tensor
    """

    mu, sigma = tf.split(1,2,nn_outputs) 

    #tf.histogram_summary('mu', mu)
    #tf.histogram_summary('sigma', sigma)
    #tf.scalar_summary('MSE',tf.reduce_mean(tf.nn.l2_loss(mu-true_outputs)))

    loss = tf.reduce_mean(tf.log(sigma)+
                          0.5*tf.square(tf.div(tf.sub(mu,true_outputs),sigma)))
    #tf.scalar_summary('loss', loss)

    # Save losses to the collection 
    tf.add_to_collection('losses',loss)

    return loss
    
def training(loss, learning_rate):
    """
    Adds to the loss model the Ops required to generate and apply gradients
    loss -- the losses tensor
    learning_rate -- learning rate for training
    """

    # create optimizer. Other options are GradientDescentOptimizer, AdagradOptimizer, etc.
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # calculate gradients
    gradients = optimizer.compute_gradients(loss)

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
