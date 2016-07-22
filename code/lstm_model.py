from __future__ import print_function
import numpy as np
import os
import sys
import time
import tensorflow as tf
import datetime as dt
import lstm_loaddatasets as ld

#===================================================================================================
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('train', False, 'When True, run training & save model. When False, load a previously saved model and evaluate it')

# Structure of the network
flags.DEFINE_integer('num_nodes', 16, 'Size of the gates')
flags.DEFINE_integer('batch_size', 50, 'Batch size')
flags.DEFINE_integer('num_unrollings', 10, 'Num unrollings')
flags.DEFINE_integer('output_vars', 10, 'Size of the output layer')
flags.DEFINE_integer('input_vars', 6, 'Size of the input layer')


# Learning rate is important for model training. 
# Decrease learning rate for more complicated models.
# Increase if convergence is steady but too slow
flags.DEFINE_float('learning_rate', 0.006, 'Initial learning rate')
flags.DEFINE_float('learning_rate_decay', 0.5, 'Learning rate decay, i.e. the fraction of the initial learning rate at the end of training')
flags.DEFINE_integer('max_steps', 15001, 'Number of steps to run trainer')

# Save models in this directory
flags.DEFINE_string('checkpoints_dir', './checkpoints', 'Directory to store checkpoints')

# Statistics
flags.DEFINE_string('summaries_dir','./logs','Summaries directory')

# Evaluation
# Output dataset
flags.DEFINE_string('output','./test_track_out.dat','When model evaluation, output the data here')
# Input dataset
flags.DEFINE_string('input','./test_track.dat','Dataset for input')
#===================================================================================================

def train():
	train_dataset, valid_dataset, test_dataset = ld.read_data_sets(num_unrollings=FLAGS.num_unrollings, batch_size=FLAGS.batch_size)
	with tf.Graph().as_default(), tf.device('/cpu:0'):

		# Create variables for input and output data moments and initialize them with train datasets' moments
		input_means = tf.get_variable('means', trainable = False, initializer = tf.convert_to_tensor(train_dataset.input_moments[0]))
		input_stds = tf.get_variable('stds', trainable = False, initializer = tf.convert_to_tensor(train_dataset.input_moments[1]))

		# Parameters:
		# Input gate: input, previous output, and bias.
		ix = tf.Variable(tf.truncated_normal([FLAGS.input_vars, FLAGS.num_nodes], -0.1, 0.1))
		im = tf.Variable(tf.truncated_normal([FLAGS.num_nodes, FLAGS.num_nodes], -0.1, 0.1))
		ib = tf.Variable(tf.zeros([1, FLAGS.num_nodes]))
		# Forget gate: input, previous output, and bias.
		fx = tf.Variable(tf.truncated_normal([FLAGS.input_vars, FLAGS.num_nodes], -0.1, 0.1))
		fm = tf.Variable(tf.truncated_normal([FLAGS.num_nodes, FLAGS.num_nodes], -0.1, 0.1))
		fb = tf.Variable(tf.zeros([1, FLAGS.num_nodes]))
		# Memory cell: input, state and bias.                             
		cx = tf.Variable(tf.truncated_normal([FLAGS.input_vars, FLAGS.num_nodes], -0.1, 0.1))
		cm = tf.Variable(tf.truncated_normal([FLAGS.num_nodes, FLAGS.num_nodes], -0.1, 0.1))
		cb = tf.Variable(tf.zeros([1, FLAGS.num_nodes]))
		# Output gate: input, previous output, and bias.
		ox = tf.Variable(tf.truncated_normal([FLAGS.input_vars, FLAGS.num_nodes], -0.1, 0.1))
		om = tf.Variable(tf.truncated_normal([FLAGS.num_nodes, FLAGS.num_nodes], -0.1, 0.1))
		ob = tf.Variable(tf.zeros([1, FLAGS.num_nodes]))
		# Variables saving state across unrollings.
		saved_output = tf.Variable(tf.zeros([FLAGS.batch_size, FLAGS.num_nodes]), trainable=False)
		saved_state = tf.Variable(tf.zeros([FLAGS.batch_size, FLAGS.num_nodes]), trainable=False)
		# Regression weights and biases.
		w = tf.Variable(tf.truncated_normal([FLAGS.num_nodes, FLAGS.output_vars], -0.1, 0.1))
		b = tf.Variable(tf.zeros([FLAGS.output_vars]))
  
		# Definition of the cell computation.
		def lstm_cell(i, o, state):
			"""Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
			Note that in this formulation, we omit the various connections between the
			previous state and the gates."""
			input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
			forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
			update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
			state = forget_gate * state + input_gate * tf.tanh(update)
			output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
			return output_gate * tf.tanh(state), state

		# Prepare placeholders for inputs and outputs
		# There is a total of 2*num_unrollings placeholders need to be fitted in the ANN
		# identified by train_inputs and train_outputs lists
		train_inputs = list()
		train_outputs = list()
		for _ in range(FLAGS.num_unrollings):
			train_inputs.append(tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.input_vars])) 
			train_outputs.append(tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.output_vars]))
    
		# Unrolled LSTM loop.
		outputs = list() # list of outputs
		output = saved_output # recall the last saved output
		state = saved_state # recall the last saved state
		for i in train_inputs:
			output, state = lstm_cell(i, output, state)
			outputs.append(output)
    
		# State saving across unrollings.
		with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
			y = tf.matmul(tf.concat(0,outputs), w)+b
			loss = tf.reduce_mean(tf.square(y - tf.concat(0,train_outputs)))
          
		# Optimizer.
		global_step = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.max_steps, FLAGS.learning_rate_decay, staircase=False)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		gradients, v = zip(*optimizer.compute_gradients(loss))
		gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
		optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
    
		# Sampling and validation eval: batch 1, no unrolling.
		sample_input = tf.placeholder(tf.float32, shape=[1,FLAGS.input_vars])
		saved_sample_output = tf.Variable(tf.zeros([1, FLAGS.num_nodes]))
		saved_sample_state = tf.Variable(tf.zeros([1, FLAGS.num_nodes]))
    
		reset_sample_state = tf.group(
 			saved_sample_output.assign(tf.zeros([1, FLAGS.num_nodes])),
			saved_sample_state.assign(tf.zeros([1, FLAGS.num_nodes])))
    
		sample_output, sample_state = lstm_cell(
			sample_input, saved_sample_output, saved_sample_state)
    
		with tf.control_dependencies([saved_sample_output.assign(sample_output), saved_sample_state.assign(sample_state)]):
			sample_prediction = tf.nn.xw_plus_b(sample_output, w, b)

		init = tf.initialize_all_variables()
        	sess = tf.Session(config = tf.ConfigProto(
			allow_soft_placement = False, # allows to utilize GPU's & CPU's
			log_device_placement = False)) # shows GPU/CPU allocation

		saver = tf.train.Saver()

        	# Finish graph creation. Below is the code for running graph
		sess.run(init)
		tf.train.start_queue_runners(sess=sess)

		#tf.initialize_all_variables().run()
		print('Initialized')
		mean_loss = 0
		num_steps = 0
		for step in range(FLAGS.max_steps):
			batches = train_dataset.next()
			feed_dict = dict()
			for i in range(FLAGS.num_unrollings):
				feed_dict[train_inputs[i]] = np.reshape(batches[i][0],(FLAGS.batch_size,FLAGS.input_vars))
				feed_dict[train_outputs[i]] = np.reshape(batches[i][1],(FLAGS.batch_size,FLAGS.output_vars))

			start_time = time.time()
			_, l, lr = sess.run([optimizer, loss, learning_rate], feed_dict=feed_dict)
			duration = time.time()-start_time

			mean_loss += l
			num_steps += 1
			if step%(FLAGS.max_steps//20) == 0:
				if step > 0:
					mean_loss = mean_loss / num_steps

				# calculate losses at validation dataset
				
				reset_sample_state.run(session = sess)
				predictions = np.zeros(shape=valid_dataset[1].shape)
				for i in range(valid_dataset[0].shape[0]):
					predictions[i] = sample_prediction.eval(
						{sample_input: np.reshape(valid_dataset[0][i,:],(1,FLAGS.input_vars))},
						session = sess)
				valid_mse = np.mean(np.square(predictions-valid_dataset[1]))
				print('Step %d (%.2f op/sec): Training MSE: %.5f, Validation MSE: %.5f' % (
			                    step, 1.0/duration, mean_loss, valid_mse))
				
				mean_loss = 0
				num_steps = 0
                 
		print('=' * 80)
		reset_sample_state.run(session = sess)
		predictions = np.zeros(shape=test_dataset[1].shape)
		for i in range(test_dataset[0].shape[0]):
        		predictions[i] = sample_prediction.eval(
				{sample_input: np.reshape(test_dataset[0][i,:],(1,FLAGS.input_vars))},
				session = sess)
		test_mse = np.mean(np.square(predictions-test_dataset[1]))
		print('Training complete. Test MSE: %.5f'%test_mse)
		for out_no in range(FLAGS.output_vars):
		        test_cc = np.corrcoef(predictions[:,out_no], test_dataset[1][:,out_no])[0,1]
			test_mse = ((predictions[:,out_no] - test_dataset[1][:,out_no]) ** 2).mean()
			print("Location %d: CC: %.4f, MSE: %.6f"%(out_no,test_cc,test_mse))

		checkpoint_path = os.path.join(FLAGS.checkpoints_dir,'model.ckpt')
		saver.save(sess, checkpoint_path, global_step=step)

#===================================================================================================
def run():
	
	# Assign datasets 
	test_ds = np.loadtxt(FLAGS.input)[:,1:7].reshape((-1, 6)).astype(np.float32)

	with tf.Graph().as_default(), tf.device('/cpu:0'):

		# Create variables for input and output data moments and initialize them with train datasets' moments
		means = tf.get_variable('means', shape=[FLAGS.input_vars], trainable = False)
		stds = tf.get_variable('stds', shape=[FLAGS.input_vars], trainable = False)

		# Parameters:
		# Input gate: input, previous output, and bias.
		ix = tf.Variable(tf.truncated_normal([FLAGS.input_vars, FLAGS.num_nodes], -0.1, 0.1))
		im = tf.Variable(tf.truncated_normal([FLAGS.num_nodes, FLAGS.num_nodes], -0.1, 0.1))
		ib = tf.Variable(tf.zeros([1, FLAGS.num_nodes]))
		# Forget gate: input, previous output, and bias.
		fx = tf.Variable(tf.truncated_normal([FLAGS.input_vars, FLAGS.num_nodes], -0.1, 0.1))
		fm = tf.Variable(tf.truncated_normal([FLAGS.num_nodes, FLAGS.num_nodes], -0.1, 0.1))
		fb = tf.Variable(tf.zeros([1, FLAGS.num_nodes]))
		# Memory cell: input, state and bias.                             
		cx = tf.Variable(tf.truncated_normal([FLAGS.input_vars, FLAGS.num_nodes], -0.1, 0.1))
		cm = tf.Variable(tf.truncated_normal([FLAGS.num_nodes, FLAGS.num_nodes], -0.1, 0.1))
		cb = tf.Variable(tf.zeros([1, FLAGS.num_nodes]))
		# Output gate: input, previous output, and bias.
		ox = tf.Variable(tf.truncated_normal([FLAGS.input_vars, FLAGS.num_nodes], -0.1, 0.1))
		om = tf.Variable(tf.truncated_normal([FLAGS.num_nodes, FLAGS.num_nodes], -0.1, 0.1))
		ob = tf.Variable(tf.zeros([1, FLAGS.num_nodes]))
		# Variables saving state across unrollings.
		saved_output = tf.Variable(tf.zeros([FLAGS.batch_size, FLAGS.num_nodes]), trainable=False)
		saved_state = tf.Variable(tf.zeros([FLAGS.batch_size, FLAGS.num_nodes]), trainable=False)
		# Regression weights and biases.
		w = tf.Variable(tf.truncated_normal([FLAGS.num_nodes, FLAGS.output_vars], -0.1, 0.1))
		b = tf.Variable(tf.zeros([FLAGS.output_vars]))

		# Definition of the cell computation.
		def lstm_cell(i, o, state):
			"""Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
			Note that in this formulation, we omit the various connections between the
			previous state and the gates."""
			input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
			forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
			update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
			state = forget_gate * state + input_gate * tf.tanh(update)
			output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
			return output_gate * tf.tanh(state), state

		# Prepare placeholders for inputs and outputs
		# There is a total of 2*num_unrollings placeholders need to be fitted in the ANN
		# identified by train_inputs and train_outputs lists
		train_inputs = list()
		train_outputs = list()
		for _ in range(FLAGS.num_unrollings):
			train_inputs.append(tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.input_vars])) 
			train_outputs.append(tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.output_vars]))
    
		# Unrolled LSTM loop.
		outputs = list() # list of outputs
		output = saved_output # recall the last saved output
		state = saved_state # recall the last saved state
		for i in train_inputs:
			output, state = lstm_cell(i, output, state)
			outputs.append(output)
    
		# State saving across unrollings.
		with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
			y = tf.matmul(tf.concat(0,outputs), w)+b
			loss = tf.reduce_mean(tf.square(y - tf.concat(0,train_outputs)))
          
		# Optimizer.
		global_step = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.max_steps, FLAGS.learning_rate_decay, staircase=False)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		gradients, v = zip(*optimizer.compute_gradients(loss))
		gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
		optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
    
		# Sampling and validation eval: batch 1, no unrolling.
		sample_input = tf.placeholder(tf.float32, shape=[1,FLAGS.input_vars])
		saved_sample_output = tf.Variable(tf.zeros([1, FLAGS.num_nodes]))
		saved_sample_state = tf.Variable(tf.zeros([1, FLAGS.num_nodes]))
    
		reset_sample_state = tf.group(
 			saved_sample_output.assign(tf.zeros([1, FLAGS.num_nodes])),
			saved_sample_state.assign(tf.zeros([1, FLAGS.num_nodes])))
    
		sample_output, sample_state = lstm_cell(
			sample_input, saved_sample_output, saved_sample_state)
    
		with tf.control_dependencies([saved_sample_output.assign(sample_output), saved_sample_state.assign(sample_state)]):
			sample_prediction = tf.nn.xw_plus_b(sample_output, w, b)

		init = tf.initialize_all_variables()
		sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = False, # allows to utilize GPU's & CPU's
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

		#tf.train.start_queue_runners(sess=sess)
		reset_sample_state.run(session = sess)
		predictions = np.zeros(shape=(test_ds.shape[0],FLAGS.output_vars))
		for i in range(test_ds.shape[0]):
        		predictions[i] = sample_prediction.eval(
				{sample_input: np.reshape(test_ds[i,:],(1,FLAGS.input_vars))},
				session = sess)
		duration = time.time()-start_time
		print('Elapsed time: %.2f sec.' % (duration))
		np.savetxt(FLAGS.output,predictions)
		print('Outputs saved as %s'%FLAGS.output)
		sess.close()

#===================================================================================================
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
