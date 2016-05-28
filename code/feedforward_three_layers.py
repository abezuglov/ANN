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


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate')
flags.DEFINE_float('learning_rate_decay', 0.1, 'Learning rate decay, i.e. the fraction of the initial learning rate at the end of training')
flags.DEFINE_integer('max_steps', 200000, 'Number of steps to run trainer')
flags.DEFINE_integer('batch_size', 25*193, 'Batch size. Divides evenly into the dataset size of 193')
flags.DEFINE_integer('hidden1', 40, 'Size of the first hidden layer')
flags.DEFINE_integer('hidden2', 20, 'Size of the second hidden layer')
flags.DEFINE_integer('hidden3', 10, 'Size of the second third layer')
flags.DEFINE_string('train_dir', './data/', 'Directory to put the training data') # not currently used
flags.DEFINE_string('checkpoints_dir', './checkpoints/', 'Directory to store checkpoints')
flags.DEFINE_string('summaries_dir','./logs/','Summaries directory')

def accuracy_mse(predictions, outputs):
    err = predictions-outputs
    return np.mean(err*err)

def placeholder_inputs(batch_size):
#Generate placeholder variables to represent input tensors
    tf_train_inputs = tf.placeholder(tf.float32, shape=(batch_size, 6)) #train_dataset2.shape(2)
    tf_train_outputs = tf.placeholder(tf.float32, shape=(batch_size, 2))
    return tf_train_inputs, tf_train_outputs

def fill_feed_dict(data_set, inputs_pl, outputs_pl):
    inputs, outputs = data_set.next_batch(FLAGS.batch_size)
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

def variable_summaries(var, name):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/'+name, mean)
    stddev = tf.reduce_mean(tf.reduce_sum(tf.square(var-mean)))
    tf.scalar_summary('stddev/'+name, stddev)
    tf.histogram_summary(name, var)

def run_training():
    train_dataset, valid_dataset, test_dataset = ld.read_data_sets()
    with tf.Graph().as_default():
        tf_train_dataset, tf_train_labels = placeholder_inputs(FLAGS.batch_size)
        tf_valid_dataset = tf.constant(valid_dataset.inputs)
        tf_test_dataset = tf.constant(test_dataset.inputs)
  
        hidden_nodes_1 = FLAGS.hidden1
        hidden_nodes_2 = FLAGS.hidden2
        hidden_nodes_3 = FLAGS.hidden3

        weights_0 = tf.Variable(tf.truncated_normal([6,hidden_nodes_1], dtype = tf.float32))
        biases_0 = tf.Variable(tf.zeros([hidden_nodes_1], dtype = tf.float32))

        weights_1 = tf.Variable(tf.truncated_normal([hidden_nodes_1,hidden_nodes_2], dtype = tf.float32))
        biases_1 = tf.Variable(tf.zeros([hidden_nodes_2], dtype = tf.float32))

        weights_2 = tf.Variable(tf.truncated_normal([hidden_nodes_2,hidden_nodes_3], dtype = tf.float32))
        biases_2 = tf.Variable(tf.zeros([hidden_nodes_3], dtype = tf.float32))

        weights_3 = tf.Variable(tf.truncated_normal([hidden_nodes_3,2], dtype = tf.float32))
        variable_summaries(weights_3, 'layer 3 weights')
        biases_3 = tf.Variable(tf.zeros([2], dtype = tf.float32))
        variable_summaries(biases_3, 'layer 3 biases')
  
        input_layer_output = tf.sigmoid(tf.matmul(tf_train_dataset, weights_0) + biases_0)
        hidden_layer_output = tf.sigmoid(tf.matmul(input_layer_output, weights_1) + biases_1)
        #hidden_layer_output = tf.nn.dropout(hidden_layer_output, 0.5)
        hidden_layer_output = tf.sigmoid(tf.matmul(hidden_layer_output, weights_2) + biases_2)
        hidden_layer_output = tf.matmul(hidden_layer_output, weights_3) + biases_3
        train_prediction = hidden_layer_output

        #loss = tf.placeholder(tf.float32)
        loss = tf.cast(tf.reduce_mean(tf.reduce_mean(tf.square(train_prediction-tf_train_labels))),tf.float32)
        
        tf.scalar_summary("Loss MSE", loss)
        global_step = tf.Variable(0.00, trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, 
                                                   global_step, FLAGS.max_steps, 
                                                   FLAGS.learning_rate_decay, staircase=False)
        
        tf.scalar_summary("Learning rate", learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        #optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
        #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
        valid_prediction = tf.sigmoid(tf.matmul(tf_valid_dataset, weights_0) + biases_0)
        valid_prediction = tf.sigmoid(tf.matmul(valid_prediction, weights_1) + biases_1)
        valid_prediction = tf.sigmoid(tf.matmul(valid_prediction, weights_2) + biases_2)
        valid_prediction = tf.matmul(valid_prediction, weights_3) + biases_3
    
        test_prediction = tf.sigmoid(tf.matmul(tf_test_dataset, weights_0) + biases_0)
        test_prediction = tf.sigmoid(tf.matmul(test_prediction, weights_1) + biases_1)
        test_prediction = tf.sigmoid(tf.matmul(test_prediction, weights_2) + biases_2)
        test_prediction = tf.matmul(test_prediction, weights_3) + biases_3

        merged = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.train.SummaryWriter(FLAGS.summaries_dir, sess.graph)
        sess.run(init)
        
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            feed_dict = fill_feed_dict(train_dataset,
                                       tf_train_dataset,
                                       tf_train_labels)
            _, l, lr, summary, predictions = sess.run([optimizer, loss, learning_rate, merged, train_prediction],feed_dict=feed_dict)
            duration = time.time()-start_time
            if (step % 10 == 0):
                summary_writer.add_summary(summary,step)
                summary_writer.flush()
            if (step % 100 == 0):
                print('Step %d: Train MSE: %.5f (%d op/sec), learning rate: %.6f' % (step, l, 1/duration, lr))
            #if (step+1)%1000 == 0 or (step+1)==FLAGS.max_steps:
            #    saver.save(sess, FLAGS.checkpoints_dir, global_step = step)
            if (step % 1000 == 0):
                print('Validation MSE: %.5f' % (do_eval(sess, valid_prediction, _, valid_dataset)))
        print('Test RMSE: %.5f' % do_eval(sess, test_prediction, _, test_dataset))
        predicted_vs_actual = np.hstack((test_prediction.eval(session = sess), test_dataset.outputs))
        print("correlation coefficients: ")
        print(np.corrcoef(predicted_vs_actual[:,0],predicted_vs_actual[:,2]))
        print(np.corrcoef(predicted_vs_actual[:,1],predicted_vs_actual[:,3]))


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
