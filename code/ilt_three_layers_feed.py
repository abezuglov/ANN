from __future__ import print_function
#import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import tensorflow as tf
import load_datasets as ld
import datetime as dt
import ilt_three_layers

flags = tf.app.flags
FLAGS = flags.FLAGS

# Learning rate is important for model training. 
# Decrease learning rate for more complicated models.
# Increase if convergence is slow but steady
flags.DEFINE_float('learning_rate', 0.5, 'Initial learning rate')
flags.DEFINE_float('learning_rate_decay', 0.1, 'Learning rate decay, i.e. the fraction of the initial learning rate at the end of training')
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer')
flags.DEFINE_float('max_loss', 0.1, 'Max acceptable validation MSE')


flags.DEFINE_integer('num_gpus',2,'Number of GPUs in the system')
flags.DEFINE_string('tower_name','ivy','Tower names')

# Split the training data into batches. Each hurricane is 193 records. Batch sizes are usually 2^k
# When batch size equals to 0, or greater than the available data, use the complete dataset
# Large batch sizes produce more accurate update gradients, but the training is slower
flags.DEFINE_integer('batch_size', 0*64*193, 'Batch size. Divides evenly into the dataset size of 193')

# Not currently used. The data is loaded in load_datasets (ld) and put in Dataset objects:
# train_dataset, valid_dataset, and test_dataset
#flags.DEFINE_string('train_dir', './data/', 'Directory to put the training data')

# Save models in this directory. TODO: save/load models
flags.DEFINE_string('checkpoints_dir', './checkpoints', 'Directory to store checkpoints')

# Statistics
flags.DEFINE_string('summaries_dir','./logs','Summaries directory')

def placeholder_inputs():
    """
    Returns placeholders for inputs and expected outputs
    """
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, FLAGS.input_vars], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, FLAGS.output_vars], name = 'y-input')
    return x, y_

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

def tower_loss(x, y_, scope):
    """
    Calculate the total loss on a single tower
    scope -- unique prefix identifying the tower
    """
    print("build graph for scope %s"%scope)
    outputs = ilt_three_layers.inference(x)
    _ = ilt_three_layers.loss(outputs, y_)
    
    losses = tf.get_collection('MSE', scope)
    total_loss = tf.add_n(losses, name='total_MSE')

    loss_avg = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_avg_op = loss_avg.apply(losses+[total_loss])

    with tf.control_dependencies([loss_avg_op]):
        total_loss = tf.identity(total_loss)
    return total_loss

def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)
        grad = tf.concat(0,grads)
        grad = tf.reduce_mean(grad,0)
        
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        x, y_ = placeholder_inputs()
        global_step = tf.get_variable(
            'global_step', [], 
            initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate, global_step, FLAGS.max_steps,
            FLAGS.learning_rate_decay, staircase=False)        

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        tower_grads = []
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d'%i):
                with tf.name_scope('%s_%d' % (FLAGS.tower_name, i)) as scope:
                    loss = tower_loss(x, y_, scope)
                    tf.get_variable_scope().reuse_variables()
                    
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step = global_step)
        merged = tf.merge_all_summaries()
           
        init = tf.initialize_all_variables()
        sess = tf.Session(config = tf.ConfigProto(
            allow_soft_placement = True, # allows to utilize GPU's & CPU's
            log_device_placement = False)) # shows GPU/CPU allocation
        # Prepare folders for saving models and its stats
        date_time_stamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir+'/train/'+date_time_stamp, sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir+'/validation/'+date_time_stamp, sess.graph)
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        train_dataset, valid_dataset, test_dataset = ld.read_data_sets()

        valid_loss = 1.0
        train_loss = 1.0
        step = 0
        while valid_loss > FLAGS.max_loss and step < FLAGS.max_steps:
            start_time = time.time()
            if step%10 != 0:
                # regular training
                feed_dict = fill_feed_dict(train_dataset, x, y_, train = True)
                
                #_, train_loss, summary = sess.run([apply_gradient_op, loss, merged], feed_dict=feed_dict)
                _, train_loss = sess.run([apply_gradient_op, loss], feed_dict=feed_dict)

                #train_writer.add_summary(summary,step)
            else:
                # check model fit
                feed_dict = fill_feed_dict(valid_dataset, x, y_, train = False)
                #valid_loss, summary = sess.run([loss, merged], feed_dict = feed_dict)
                valid_loss = sess.run([loss], feed_dict = feed_dict)
                #test_writer.add_summary(summary,step)
                duration = time.time()-start_time
                print('Step %d (%d op/sec): Training MSE: %.5f, Validation MSE: %.5f' % (
                    step, 1/duration, np.float32(train_loss).item(), np.float32(valid_loss).item()))
                #print('Step %d (%d op/sec): Training MSE: %.5f, Validation MSE: %.5f' % (step, 1/duration, 0,0))
            step+=1
            
        feed_dict = fill_feed_dict(test_dataset, x, y_, train = False)
        test_loss = sess.run([loss], feed_dict = feed_dict)
        print('Test MSE: %.5f' % (np.float32(test_loss).item()))



def run_training():
    """
    Run training with parameters and graph from ilt_three_layers
    """
    train_dataset, valid_dataset, test_dataset = ld.read_data_sets()

    with tf.Graph().as_default():
        x, y_ = placeholder_inputs()

        MSE = ilt_three_layers.loss(ilt_three_layers.inference(x), y_)
        tf.scalar_summary('MSE', MSE)
        
        apply_gradient_op = ilt_three_layers.training(MSE, FLAGS.learning_rate)

        merged = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess = tf.Session(config = tf.ConfigProto(
            allow_soft_placement = True, # allows to utilize GPU's & CPU's
            log_device_placement = False)) # shows GPU/CPU allocation

        # Prepare folders for saving models and its stats
        date_time_stamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir+'/train/'+date_time_stamp, sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir+'/validation/'+date_time_stamp, sess.graph)

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
                _, train_loss, summary = sess.run([apply_gradient_op, MSE, merged], feed_dict=feed_dict)
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
    #run_training()
    train()

if __name__ == "__main__":
    main(sys.argv)
