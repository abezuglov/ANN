from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf
import load_datasets as ld
import datetime as dt

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate')
flags.DEFINE_float('learning_rate_decay', 0.1, 'Learning rate decay, i.e. the fraction of the initial learning rate at the end of training')

flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer')
flags.DEFINE_float('max_loss', 0.1, 'Max acceptable validation MSE')
flags.DEFINE_integer('batch_size', 64*193, 'Batch size. Divides evenly into the dataset size of 193')
flags.DEFINE_integer('hidden1', 15, 'Size of the first hidden layer')
flags.DEFINE_integer('hidden2', 8, 'Size of the second hidden layer')
flags.DEFINE_integer('hidden3', 3, 'Size of the third hidden layer')
flags.DEFINE_integer('output_vars', 2, 'Size of the output layer')
flags.DEFINE_integer('input_vars', 6, 'Size of the input layer')
#flags.DEFINE_string('train_dir', './data/', 'Directory to put the training data') # not currently used
flags.DEFINE_string('checkpoints_dir', './checkpoints/three-layer/'+dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Directory to store checkpoints')
flags.DEFINE_string('summaries_dir','./logs/three-layer/'+dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'Summaries directory')

def weight_variable(name, shape):
    """
    Returns TF weight variable with given shape. The weights are normally distributed with mean = 0, stddev = 0.1
    shape -- shape of the variable, i.e. [4,5] matrix of 4x5
    """
    with tf.device('/cpu:0'):
        initial = tf.truncated_normal_initializer(stddev = 0.1)
        var = tf.get_variable(name, shape, initializer = initial)
    return var

def bias_variable(name, shape):
    """
    Returns TF bias variable with given shape. The biases are initially at 0.1
    shape -- shape of the variable, i.e. [4] -- vector of length 4
    """
    with tf.device('/cpu:0'):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable(name, shape, initializer = initial)
    return var

def variable_summaries(var, name):
    """
    Adds multiple summaries (statistics) for a TF variable
    var -- TF variable
    name -- variable name
    """
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/'+name, mean)
    stddev = tf.reduce_mean(tf.reduce_sum(tf.square(var-mean)))
    tf.scalar_summary('stddev/'+name, stddev)
    _min = tf.reduce_min(var)
    tf.scalar_summary('min/'+name, _min)
    _max = tf.reduce_max(var)
    tf.scalar_summary('max/'+name, _max)
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
    hidden_1 = nn_layer(inputs, FLAGS.input_vars, FLAGS.hidden1, 'layer1')
    hidden_2 = nn_layer(hidden_1, FLAGS.hidden1, FLAGS.hidden2, 'layer2')
    hidden_3 = nn_layer(hidden_2, FLAGS.hidden2, FLAGS.hidden3, 'layer3')      
    train_prediction = nn_layer(hidden_3, FLAGS.hidden3, FLAGS.output_vars, 'output', act = None)      
    return train_prediction

def loss(nn_outputs, true_outputs):
    prediction_diff = nn_outputs-true_outputs
    MSE = tf.cast(tf.reduce_mean(tf.reduce_mean(tf.square(prediction_diff))),tf.float32)
    return MSE
    
def training(MSE, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(MSE)
    global_step = tf.Variable(0.00, trainable=False)
    apply_gradient_op = optimizer.apply_gradients(gradients, global_step = global_step)
    return apply_gradient_op

def evaluation(nn_outputs, true_outputs):
    return loss(nn_outputs, true_outputs)
