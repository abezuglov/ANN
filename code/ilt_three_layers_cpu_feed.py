from __future__ import print_function
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

flags.DEFINE_boolean('train', False, 'When True, run training & save model. When False, load a previously saved model and evaluate it')

# Learning rate is important for model training. 
# Decrease learning rate for more complicated models.
# Increase if convergence is slow but steady
flags.DEFINE_float('learning_rate', 0.05, 'Initial learning rate')
flags.DEFINE_float('learning_rate_decay', 0.1, 'Learning rate decay, i.e. the fraction of the initial learning rate at the end of training')
flags.DEFINE_integer('max_steps', 200, 'Number of steps to run trainer')
flags.DEFINE_float('max_loss', 0.1, 'Max acceptable validation MSE')
flags.DEFINE_float('moving_avg_decay', 0.999, 'Moving average decay for training variables')

# Split the training data into batches. Each hurricane is 193 records. Batch sizes are usually 2^k
# When batch size equals to 0, or exceeds available data, use the whole dataset
# Large batch sizes produce more accurate update gradients, but the training is slower
flags.DEFINE_integer('batch_size', 64*193, 'Batch size. Divides evenly into the dataset size of 193')

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
    train -- if TRUE, then return DS in batches for training. Otherwise, return complete DS for validation/testing
    """
    if train:
        batch_size = FLAGS.batch_size
    else:
        batch_size = 0

    # Read next batch of data from the dataset
    inputs, outputs = data_set.next_batch(batch_size = batch_size)

    # Create dictionary for return
    feed_dict = {
        inputs_pl: inputs,
        outputs_pl: outputs
    }
    return feed_dict

def train():
    """
    Finish building the graph and run training on a single CPU's
    """
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        x, y_ = placeholder_inputs()
        global_step = tf.get_variable(
            'global_step', [], 
            initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate, global_step, FLAGS.max_steps,
            FLAGS.learning_rate_decay, staircase=False)        

        # create a standard gradient descent optimizer
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        outputs = ilt_three_layers.inference(x)
        loss = ilt_three_layers.loss(outputs, y_)
        tf.scalar_summary('MSE', loss)

        grads = optimizer.compute_gradients(loss)
        
        # apply the gradients to the model
        apply_gradient_op = optimizer.apply_gradients(grads, global_step = global_step)

        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_avg_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op, variables_averages_op)
        #train_op = apply_gradient_op

        merged = tf.merge_all_summaries()
           
        init = tf.initialize_all_variables()
        sess = tf.Session(config = tf.ConfigProto(
            allow_soft_placement = False, # allows to utilize GPU's & CPU's
            log_device_placement = False)) # shows GPU/CPU allocation
        # Prepare folders for saving models and its stats
        date_time_stamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir+'/train/'+date_time_stamp, sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir+'/validation/'+date_time_stamp, sess.graph)
        saver = tf.train.Saver()

        # Below is the code for running graph
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        # Assign datasets 
        train_dataset, valid_dataset, test_dataset = ld.read_data_sets()

        valid_loss = 1.0
        train_loss = 1.0
        step = 1
        # Main training loop
        while valid_loss > FLAGS.max_loss and step < FLAGS.max_steps:
            start_time = time.time()
            if step%100 != 0:
                # regular training
                feed_dict = fill_feed_dict(train_dataset, x, y_, train = True)
                
                _, train_loss, summary, lr = sess.run([train_op, loss, merged, learning_rate], feed_dict=feed_dict)
                duration = time.time()-start_time
                train_writer.add_summary(summary,step)
            else:
                # check model fit
                feed_dict = fill_feed_dict(valid_dataset, x, y_, train = False)
                valid_loss, summary = sess.run([loss, merged], feed_dict = feed_dict)
                test_writer.add_summary(summary,step)
                duration = time.time()-start_time
                print('Step %d (%.2f op/sec): Training MSE: %.5f, Validation MSE: %.5f' % (
                    step, 1.0/duration, np.float32(train_loss).item(), np.float32(valid_loss).item()))
            #if step%1000 != 0:
            #    checkpoint_path = os.path.join(FLAGS.checkpoints_dir,'model.ckpt')
            #    saver.save(sess, checkpoint_path, global_step=step)
            step+=1

        checkpoint_path = os.path.join(FLAGS.checkpoints_dir,'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

        feed_dict = fill_feed_dict(test_dataset, x, y_, train = False)
        test_loss = sess.run([loss], feed_dict = feed_dict)
        print('Test MSE: %.5f' % (np.float32(test_loss).item()))
        sess.close()

def run():
    """
    Finish building the graph and run it on a single CPU's
    """
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        x, y_ = placeholder_inputs()
        outputs = ilt_three_layers.inference(x)
        loss = ilt_three_layers.loss(outputs, y_)

        init = tf.initialize_all_variables()
        sess = tf.Session(config = tf.ConfigProto(
            allow_soft_placement = False, # allows to utilize GPU's & CPU's
            log_device_placement = False)) # shows GPU/CPU allocation
         
        start_time = time.time()
        # Below is the code for running graph
        sess.run(init)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoints_dir)
        if ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model %s restored"%ckpt.model_checkpoint_path)
        else:
            print("Could not find any checkpoints at %s"%FLAGS.checkpoints_dir)
            return

        tf.train.start_queue_runners(sess=sess)

        # Assign datasets 
        train_dataset, valid_dataset, test_dataset = ld.read_data_sets()
        feed_dict = fill_feed_dict(train_dataset, x, y_, train = False)
        test_loss = sess.run([loss], feed_dict = feed_dict)
        duration = time.time()-start_time
        print('Elapsed time: %.2f sec. Test MSE: %.5f' % (duration, np.float32(test_loss).item()))
        sess.close()

    
def main(argv):
    if(FLAGS.train):
        if tf.gfile.Exists(FLAGS.summaries_dir):
            tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
        tf.gfile.MakeDirs(FLAGS.summaries_dir)
        if tf.gfile.Exists(FLAGS.checkpoints_dir):
            tf.gfile.DeleteRecursively(FLAGS.checkpoints_dir)
        tf.gfile.MakeDirs(FLAGS.checkpoints_dir)
        train()
    else:
        run()
    
if __name__ == "__main__":
    main(sys.argv)
