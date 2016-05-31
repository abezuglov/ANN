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

flags.DEFINE_float('learning_rate', 0.08, 'Initial learning rate')
flags.DEFINE_float('learning_rate_decay', 0.1, 'Learning rate decay, i.e. the fraction of the initial learning rate at the end of training')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer')
flags.DEFINE_integer('batch_size', 50*193, 'Batch size. Divides evenly into the dataset size of 193')
flags.DEFINE_integer('hidden1', 15, 'Size of the first hidden layer')
flags.DEFINE_integer('hidden2', 8, 'Size of the second hidden layer')
flags.DEFINE_integer('hidden3', 3, 'Size of the third hidden layer')
flags.DEFINE_integer('output_vars', 2, 'Size of the output layer')
flags.DEFINE_integer('input_vars', 6, 'Size of the input layer')
#flags.DEFINE_string('train_dir', './data/', 'Directory to put the training data') # not currently used
flags.DEFINE_string('checkpoints_dir', './checkpoints/three-layer/'+dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Directory to store checkpoints')
flags.DEFINE_string('summaries_dir','./logs/three-layer/'+dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'Summaries directory')

def accuracy_mse(predictions, outputs):
    err = predictions-outputs
    return np.mean(err*err)

def placeholder_inputs(batch_size):
#Generate placeholder variables to represent input tensors
    if batch_size is None:
        tf_train_inputs = tf.placeholder(tf.float32, shape=(None, 6)) #train_dataset2.shape(2)
        tf_train_outputs = tf.placeholder(tf.float32, shape=(None, 2))
    else:
        tf_train_inputs = tf.placeholder(tf.float32, shape=(batch_size, 6)) #train_dataset2.shape(2)
        tf_train_outputs = tf.placeholder(tf.float32, shape=(batch_size, 2))
    return tf_train_inputs, tf_train_outputs

def fill_feed_dict(data_set, inputs_pl, outputs_pl):
    inputs, outputs = data_set.get_full()#next_batch(FLAGS.batch_size)
    feed_dict = {
        inputs_pl: inputs,
        outputs_pl: outputs
    }
    return feed_dict

# Runs one evaluation against full epoch of data
def do_eval(sess,
            inputs_pl,
            outputs_pl,
            data_set):
    #steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    #num_examples = steps_per_epoch * FLAGS.batch_size
    #for step in xrange(steps_per_epoch):
    #    feed_dict = fill_feed_dict(data_set,
    #                               inputs_pl,
    #                               outputs_pl)
    #    #sess.run(feed_dict = feed_dict)
    #    print("do_eval run: %.6f" % accuracy_mse(data_set.inputs.eval(session = sess), data_set.outputs))
    return accuracy_mse(inputs_pl.eval(session = sess), data_set.outputs)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def variable_summaries(var, name):
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
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights, layer_name+'/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
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
    train_dataset, valid_dataset, test_dataset = ld.read_data_sets()
    with tf.Graph().as_default():
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, FLAGS.input_vars], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, FLAGS.output_vars], name = 'y-input')
        #tf_valid_dataset = tf.constant(valid_dataset.inputs)
        #tf_test_dataset = tf.constant(test_dataset.inputs)
  
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
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
            #optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
                MSE, global_step=global_step)
                  
        merged = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.train.SummaryWriter(FLAGS.summaries_dir, sess.graph)
        sess.run(init)
        
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            feed_dict = fill_feed_dict(train_dataset, x, y_)
            _, train_loss, lr, summary = sess.run([optimizer, MSE, learning_rate, merged], feed_dict=feed_dict)
            duration = time.time()-start_time
            #print('Step %d: Train MSE: %.5f (%d op/sec), learning rate: %.6f' % (step, loss, 1/duration, lr))

            if step%10 == 0:
                feed_dict = fill_feed_dict(valid_dataset, x, y_)
                valid_loss, summary = sess.run([MSE, merged], feed_dict = feed_dict)
                print('Step %d (%d op/sec): Train MSE: %.5f, Validation MSE: %.5f' % (step, 1/duration, train_loss, valid_loss))
 
            summary_writer.add_summary(summary,step)
            summary_writer.flush()
            
        feed_dict = fill_feed_dict(test_dataset, x, y_)
        test_loss, summary = sess.run([MSE, merged], feed_dict = feed_dict)
        print('Test MSE: %.5f' % (test_loss))
        
        #predicted_vs_actual = np.hstack((test_prediction.eval(session = sess), test_dataset.outputs))
        #print("correlation coefficients: ")
        #print(np.corrcoef(predicted_vs_actual[:,0],predicted_vs_actual[:,2]))
        #print(np.corrcoef(predicted_vs_actual[:,1],predicted_vs_actual[:,3]))


def main(argv):
    #if tf.gfile.Exists(FLAGS.summaries_dir):
    #    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    #tf.gfile.MakeDirs(FLAGS.summaries_dir)
    #if tf.gfile.Exists(FLAGS.checkpoints_dir):
    #    tf.gfile.DeleteRecursively(FLAGS.checkpoints_dir)
    #tf.gfile.MakeDirs(FLAGS.checkpoints_dir)

    run_training()

if __name__ == "__main__":
    main(sys.argv)
