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

flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer')
flags.DEFINE_float('max_loss', 0.1, 'Max acceptable validation MSE')
flags.DEFINE_integer('batch_size', 128*193, 'Batch size. Divides evenly into the dataset size of 193')
flags.DEFINE_integer('hidden1', 15, 'Size of the first hidden layer')
flags.DEFINE_integer('hidden2', 8, 'Size of the second hidden layer')
flags.DEFINE_integer('hidden3', 3, 'Size of the third hidden layer')
flags.DEFINE_integer('output_vars', 2, 'Size of the output layer')
flags.DEFINE_integer('input_vars', 6, 'Size of the input layer')
#flags.DEFINE_string('train_dir', './data/', 'Directory to put the training data') # not currently used
flags.DEFINE_string('checkpoints_dir', './checkpoints/three-layer/'+dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Directory to store checkpoints')
flags.DEFINE_string('summaries_dir','./logs/three-layer/'+dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'Summaries directory')

def fill_feed_dict(data_set, inputs_pl, outputs_pl, train):
    """
    Returns feed dictionary for TF. 
    data_set -- dataset
    inputs_pl -- TF placeholder for inputs
    outputs_pl -- TF placeholder for outputs
    train -- if TRUE, then return DS in batches for training. Otherwise, return the complete DS for validation/testing
    """
    if train:
        batch_size = FLAGS.batch_size
    else:
        batch_size = 0

    inputs, outputs = data_set.next_batch(batch_size = batch_size)
    feed_dict = {
        inputs_pl: inputs,
        outputs_pl: outputs
    }
    return feed_dict

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

def run_training():
    """
    Creates a NN and runs its training/running
    """
    train_dataset, valid_dataset, test_dataset = ld.read_data_sets()
    with tf.Graph().as_default():
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, FLAGS.input_vars], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, FLAGS.output_vars], name = 'y-input')
  
        hidden_1 = nn_layer(x, FLAGS.input_vars, FLAGS.hidden1, 'layer1')
        hidden_2 = nn_layer(hidden_1, FLAGS.hidden1, FLAGS.hidden2, 'layer2')
        hidden_3 = nn_layer(hidden_2, FLAGS.hidden2, FLAGS.hidden3, 'layer3')      
        train_prediction = nn_layer(hidden_3, FLAGS.hidden3, FLAGS.output_vars, 'output', act = None)      
        
        with tf.name_scope('MSE'):
            prediction_diff = train_prediction-y_
            MSE = tf.cast(tf.reduce_mean(tf.reduce_mean(tf.square(prediction_diff))),tf.float32)
            tf.scalar_summary('MSE', MSE)

        with tf.name_scope('train'):
            global_step = tf.Variable(0.00, trainable=False)
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, 
                                                       global_step, FLAGS.max_steps, 
                                                       FLAGS.learning_rate_decay, staircase=False)        
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients = optimizer.compute_gradients(MSE)
            apply_gradient_op = optimizer.apply_gradients(gradients, global_step = global_step)
                  
        merged = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess = tf.Session(config = tf.ConfigProto(
            allow_soft_placement = True,
            log_device_placement = False))
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir+'/train', sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir+'/validation', sess.graph)
        sess.run(init)
        
        #for step in xrange(FLAGS.max_steps):
        valid_loss = 1.0
        train_loss = 1.0
        step = 0
        while valid_loss > FLAGS.max_loss and step < FLAGS.max_steps:
            start_time = time.time()
            if step%100 != 0:
                # regular training
                feed_dict = fill_feed_dict(train_dataset, x, y_, train = True)
                _, train_loss, lr, summary = sess.run([apply_gradient_op, MSE, learning_rate, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary,step)
            else:
                # check model fit
                feed_dict = fill_feed_dict(valid_dataset, x, y_, train = False)
                valid_loss, summary = sess.run([MSE, merged], feed_dict = feed_dict)
                test_writer.add_summary(summary,step)
                duration = time.time()-start_time
                print('Step %d (%d op/sec): Training MSE: %.5f, Validation MSE: %.5f' % (step, 1/duration, train_loss, valid_loss))
            step+=1
            
        feed_dict = fill_feed_dict(test_dataset, x, y_, train = False)
        test_loss, summary = sess.run([MSE, merged], feed_dict = feed_dict)
        print('Test MSE: %.5f' % (test_loss))
        
        #predicted_vs_actual = np.hstack((test_prediction.eval(session = sess), test_dataset.outputs))
        #print("correlation coefficients: ")
        #print(np.corrcoef(predicted_vs_actual[:,0],predicted_vs_actual[:,2]))
        #print(np.corrcoef(predicted_vs_actual[:,1],predicted_vs_actual[:,3]))


def main(argv):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    if tf.gfile.Exists(FLAGS.checkpoints_dir):
        tf.gfile.DeleteRecursively(FLAGS.checkpoints_dir)
    tf.gfile.MakeDirs(FLAGS.checkpoints_dir)

    run_training()

if __name__ == "__main__":
    main(sys.argv)
