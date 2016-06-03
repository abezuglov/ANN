from __future__ import print_function
import numpy as np
import os
import sys
import time
import tensorflow as tf
import datetime as dt

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('hidden1', 15, 'Size of the first hidden layer')
flags.DEFINE_integer('hidden2', 8, 'Size of the second hidden layer')
flags.DEFINE_integer('hidden3', 3, 'Size of the third hidden layer')
flags.DEFINE_integer('output_vars', 2, 'Size of the output layer')
flags.DEFINE_integer('input_vars', 6, 'Size of the input layer')

def weight_variable(name, shape):
    """
    Returns a shared TF weight variable with given shape. The weights are normally distributed with mean = 0, stddev = 0.1
    shape -- shape of the variable, i.e. [4,5] matrix of 4x5
    """
    with tf.device('/cpu:0'):
        initial = tf.truncated_normal_initializer(stddev = 0.1)
        var = tf.get_variable(name, shape, initializer = initial)
    return var

def bias_variable(name, shape):
    """
    Returns a shared TF bias variable with given shape. The biases are initially at 0.1
    shape -- shape of the variable, i.e. [4] -- vector of length 4
    """
    with tf.device('/cpu:0'):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable(name, shape, initializer = initial)
    return var

def variable_summaries(var, name):
    """
    Adds multiple (excessive) summaries (statistics) for a TF variable
    var -- TF variable
    name -- variable name
    """
    mean = tf.reduce_mean(var)
    tf.scalar_summary(name+'/mean', mean)
    stddev = tf.reduce_mean(tf.reduce_sum(tf.square(var-mean)))
    tf.scalar_summary(name+'/stddev', stddev)
    #_min = tf.reduce_min(var)
    #tf.scalar_summary(name+'/min', _min)
    #_max = tf.reduce_max(var)
    #tf.scalar_summary(name+'/max', _max)
    tf.histogram_summary(name, var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act = tf.sigmoid):
    """
    Creates and returns NN layer
    input_tensor -- TF tensor at layer input
    input_dim -- size of layer input
    output_dim -- size of layer output
    layer_name -- name of the layer for summaries (statistics)
    act -- nonlinear activation function
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable(layer_name+'/weights',[input_dim, output_dim])
            variable_summaries(weights, layer_name+'/weights')
        with tf.name_scope('biases'):
            biases = bias_variable(layer_name+'/biases',[output_dim])
            variable_summaries(biases, layer_name+'/biases')
        with tf.name_scope('WX_plus_b'):
            preactivate = tf.matmul(input_tensor, weights)+biases
            tf.histogram_summary(layer_name+'/pre_activations', preactivate)
        if act is not None:
            activations = act(preactivate, 'activation')
        else:
            activations = preactivate
        tf.histogram_summary(layer_name+'/activations', activations)
    return activations


def inference(inputs):
    """
    Build the graph/ANN to make predictions.
    inputs -- tensor representing network inputs
    train_prediction (return) -- network outputs
    """
    hidden_1 = nn_layer(inputs, FLAGS.input_vars, FLAGS.hidden1, 'layer1')
    hidden_2 = nn_layer(hidden_1, FLAGS.hidden1, FLAGS.hidden2, 'layer2')
    hidden_3 = nn_layer(hidden_2, FLAGS.hidden2, FLAGS.hidden3, 'layer3')      
    train_prediction = nn_layer(hidden_3, FLAGS.hidden3, FLAGS.output_vars, 'output', act = None)      
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
    print("Gradients:")
    print(gradients)

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
