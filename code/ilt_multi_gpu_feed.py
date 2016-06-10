from __future__ import print_function
import numpy as np
import os
import sys
import time
import tensorflow as tf
import load_datasets as ld
import datetime as dt
import ilt_one_layer as ilt

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('train', True, ' If True, run training & save model, otherwise -- load a previously saved model and evaluate it')

# Multi-GPU settings
flags.DEFINE_integer('num_gpus',2,'Number of GPUs in the system')
flags.DEFINE_string('tower_name','ivy','Tower names')

# Split the training data into batches. Each hurricane is 193 records. Batch sizes are usually 2^k
# When batch size equals to 0, or exceeds available data, use the whole dataset
# Large batch sizes produce more accurate update gradients, but the training is slower
flags.DEFINE_integer('batch_size', 64*193, 'Batch size. Divides evenly into the dataset size of 193')

# Not currently used. The data is loaded in load_datasets (ld) and put in Dataset objects:
# train_dataset, valid_dataset, and test_dataset
#flags.DEFINE_string('train_dir', './data/', 'Directory to put the training data')

# Save models in this directory
flags.DEFINE_string('checkpoints_dir', './checkpoints', 'Directory to store checkpoints')

# Statistics
flags.DEFINE_string('summaries_dir','./logs','Summaries directory')

# Evaluation
# Output dataset
flags.DEFINE_string('output','./test_track_out.dat','When model evaluation, output the data here')
# Input dataset
flags.DEFINE_string('input','./test_track.dat','Dataset for input')

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

def tower_loss(x, y_, scope):
    """
    Calculate the total loss on a single tower
    x, y_ -- inputs and expected outputs
    scope -- unique prefix identifying the tower
    
    Note: The graph is created on /cpu:0. The code below reuses the graph
    """
    # Run inference and calculate the losses. The losses are stored in the collection
    # so skip the returns
    outputs = ilt.inference(x)
    _ = ilt.loss(outputs, y_)
    
    # Read the losses from the collection and sum them up
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')

    loss_avg = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, name='avg')
    loss_avg_op = loss_avg.apply(losses+[total_loss])

    with tf.control_dependencies([loss_avg_op]):
        total_loss = tf.identity(total_loss)

    return total_loss

def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers
    tower_grads -- list of lists of tuples (gradient, variable)
    """
    average_grads = []

    # zip(*tower_grads) puts grads for each variable together
    # grad_and_vars is a tuple of tuples ((grad_gpu1, var1),(grad_gpu2, var1))
    for grad_and_vars in zip(*tower_grads):
        grads = []
        # g each individual gradient
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)
        # grad average gradient across the gpu's
        grad = tf.concat(0,grads)
        grad = tf.reduce_mean(grad,0)
        
        # get the variable as the second element from the first tuple
        v = grad_and_vars[0][1]
        # combine the gradient and append it to the average_grads
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train():
    """
    Build the graph and run training on multiple GPU's
    """
    # Assign datasets 
    train_dataset, valid_dataset, test_dataset = ld.read_data_sets()

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Prepare placeholders for inputs and expected outputs
        x = tf.placeholder(tf.float32, [None, FLAGS.input_vars], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, FLAGS.output_vars], name = 'y-input')

        # Create variables for input data moments and initialize them with train datasets' moments
        means = tf.get_variable('means', trainable = False, 
                                initializer = tf.convert_to_tensor(train_dataset.means))
        stds = tf.get_variable('stds', trainable = False, 
                                initializer = tf.convert_to_tensor(train_dataset.stds))

        # Normalize input data
        x_normalized = tf.div(tf.sub(x,means),stds)

        global_step = tf.get_variable(
            'global_step', [], 
            initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate, global_step, FLAGS.max_steps,
            FLAGS.learning_rate_decay, staircase=False)        

        # create a standard gradient descent optimizer
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # tower_grads -- list of gradients (list of list of tuples like (grad1, var1))
        tower_grads = []
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d'%i): # make sure TF runs the code on the GPU:%d tower
                with tf.name_scope('%s_%d' % (FLAGS.tower_name, i)) as scope:
                    # Construct the entire ANN, but share the vars across the towers
                    loss = tower_loss(x_normalized, y_, scope)
                   
                    # Make sure that the vars are reused for the next tower
                    tf.get_variable_scope().reuse_variables()

                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    
                    # calculate the gradients and add them to the list
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

        summaries.append(tf.scalar_summary('MSE',loss))
        summaries.append(tf.scalar_summary('CC',tf.get_collection('cc')[0]))

        # calculate average gradients
        grads = average_gradients(tower_grads)

        #for grad, var in grads:
            #if grad is not None:
            #summaries.append(tf.histogram_summary(var.op.name+'/gradients', grad))
            #summaries.append(tf.scalar_summary(var.op.name+'/sparsity',tf.nn.zero_fraction(var)))

        # apply the gradients to the model
        apply_gradient_op = optimizer.apply_gradients(grads, global_step = global_step)

        # you wanna run the code slower?
        #for var in tf.trainable_variables():
        #    summaries.append(tf.histogram_summary(var.op.name, var))

        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_avg_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op, variables_averages_op)
        #train_op = apply_gradient_op

        merged = tf.merge_all_summaries()
           
        init = tf.initialize_all_variables()
        sess = tf.Session(config = tf.ConfigProto(
            allow_soft_placement = True, # allows to utilize GPU's & CPU's
            log_device_placement = False)) # shows GPU/CPU allocation
        # Prepare folders for saving models and its stats
        date_time_stamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir+'/train/'+date_time_stamp) #,sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir+'/validation/'+date_time_stamp)
        saver = tf.train.Saver(tf.all_variables())

        # Below is the code for running graph
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        valid_loss = 1.0
        train_loss = 1.0
        step = 1
        # Main training loop
        while valid_loss > FLAGS.max_loss and step < FLAGS.max_steps:
            start_time = time.time()
            # regular training
            feed_dict = fill_feed_dict(train_dataset, x, y_, train = True)
                
            _, train_loss, summary, lr = sess.run([train_op, loss, merged, learning_rate], feed_dict=feed_dict)
            duration = time.time()-start_time
            train_writer.add_summary(summary,step)
            if step%(FLAGS.max_steps//20) == 0:
                # check model fit
                feed_dict = fill_feed_dict(valid_dataset, x, y_, train = False)
                valid_loss, summary = sess.run([loss, merged], feed_dict = feed_dict)
                test_writer.add_summary(summary,step)
                duration = time.time()-start_time
                
                print('Step %d (%.2f op/sec): Training MSE: %.5f, Validation MSE: %.5f' % (
                    step, 1.0/duration, np.float32(train_loss).item(), np.float32(valid_loss).item()))
            step+=1

        checkpoint_path = os.path.join(FLAGS.checkpoints_dir,'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
            
        feed_dict = fill_feed_dict(test_dataset, x, y_, train = False)
        test_loss = sess.run([loss], feed_dict = feed_dict)
        print('Test MSE: %.5f' % (np.float32(test_loss).item()))
        sess.close()

def run():
    """
    Finish building the graph and run it at the default device (CPU or GPU)
    """
    # Assign datasets 
    test_ds = np.loadtxt(FLAGS.input)[:,1:7].reshape((-1, 6)).astype(np.float32)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Prepare placeholders for inputs and expected outputs
        x = tf.placeholder(tf.float32, [None, FLAGS.input_vars], name='x-input')

        means = tf.get_variable('means', shape=[FLAGS.input_vars], trainable = False)
        stds = tf.get_variable('stds', shape=[FLAGS.input_vars], trainable = False)

        # Normalize input data
        x_normalized = tf.div(tf.sub(x,means),stds)

        outputs = ilt.inference(x_normalized)

        init = tf.initialize_all_variables()
        sess = tf.Session(config = tf.ConfigProto(
            allow_soft_placement = False, # allows to utilize GPU's & CPU's
            log_device_placement = False)) # shows GPU/CPU allocation
         
        start_time = time.time()
        # Below is the code for running graph
        sess.run(init)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoints_dir)
        if ckpt != None and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model %s restored"%ckpt.model_checkpoint_path)
        else:
            print("Could not find any checkpoints at %s"%FLAGS.checkpoints_dir)
            return

        tf.train.start_queue_runners(sess=sess)
        
        out = sess.run(outputs, feed_dict = {x:test_ds})
        duration = time.time()-start_time
        print('Elapsed time: %.2f sec.' % (duration))
        np.savetxt(FLAGS.output,out)
        print('Outputs saved as %s'%FLAGS.output)
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
        if tf.gfile.Exists(FLAGS.output+'*'):
            tf.gfile.DeleteRecursively(FLAGS.output+'*')
        run()

if __name__ == "__main__":
    main(sys.argv)