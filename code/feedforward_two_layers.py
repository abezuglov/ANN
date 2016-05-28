from __future__ import print_function
#import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf
import load_datasets as d

def accuracy_mse(predictions, outputs):
    err = predictions-outputs
    return np.mean(err*err)

# Deep ANN
batch_size = 20*193
hidden_nodes_1 = 40
hidden_nodes_2 = 25
hidden_nodes_3 = 10

num_steps = 1000001
starter_learning_rate = 0.05
rate_decay = 0.1

train_dataset2 = d.train_dataset2
train_output = d.train_output
valid_dataset2 = d.valid_dataset2
valid_output = d.valid_output
test_dataset2 = d.test_dataset2
test_output = d.test_output

graph = tf.Graph()
with graph.as_default():

    # Input data.
    # Load the training, validation and test data into constants that are
    # attached to the graph.
    
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 6)) #train_dataset2.shape(2)
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 2))
    tf_valid_dataset = tf.constant(valid_dataset2)
    tf_test_dataset = tf.constant(test_dataset2)
  
    weights_0 = tf.Variable(tf.truncated_normal([6,hidden_nodes_1], dtype = tf.float32))
    biases_0 = tf.Variable(tf.zeros([hidden_nodes_1], dtype = tf.float32))
    
    weights_1 = tf.Variable(tf.truncated_normal([hidden_nodes_1,hidden_nodes_2], dtype = tf.float32))
    biases_1 = tf.Variable(tf.zeros([hidden_nodes_2], dtype = tf.float32))
    
    weights_2 = tf.Variable(tf.truncated_normal([hidden_nodes_2,2], dtype = tf.float32))
    biases_2 = tf.Variable(tf.zeros([2], dtype = tf.float32))

  
    input_layer_output = tf.sigmoid(tf.matmul(tf_train_dataset, weights_0) + biases_0)
    hidden_layer_output = tf.sigmoid(tf.matmul(input_layer_output, weights_1) + biases_1)
    #hidden_layer_output = tf.nn.dropout(hidden_layer_output, 0.5)
    hidden_layer_output = tf.matmul(hidden_layer_output, weights_2) + biases_2
    
    
    loss = tf.cast(tf.reduce_mean(tf.reduce_mean(tf.square(hidden_layer_output-tf_train_labels))),tf.float32)
        
    global_step = tf.Variable(0.00, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, num_steps, rate_decay, staircase=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
    train_prediction = loss
    valid_prediction = tf.sigmoid(tf.matmul(tf_valid_dataset, weights_0) + biases_0)
    valid_prediction = tf.sigmoid(tf.matmul(valid_prediction, weights_1) + biases_1)
    valid_prediction = tf.matmul(valid_prediction, weights_2) + biases_2
    
    test_prediction = tf.sigmoid(tf.matmul(tf_test_dataset, weights_0) + biases_0)
    test_prediction = tf.sigmoid(tf.matmul(test_prediction, weights_1) + biases_1)
    test_prediction = tf.matmul(test_prediction, weights_2) + biases_2

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_output.shape[0] - batch_size)
        batch_data = train_dataset2[offset:(offset + batch_size), :]
        batch_output = train_output[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_output}
        _, l, lr, predictions = session.run([optimizer, loss, learning_rate, train_prediction],feed_dict=feed_dict)
        if (step % 1000 == 0):
            print('Loss at step %d: %f; learning rate: %.6f' % (step, l, lr))
        if (step % 10000 == 0):
            # print('Training MSE: %.4f' % accuracy_mse(predictions, train_output))
            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph
            # dependencies.
            print('Validation at step %d MSE: %.4f' % (step,accuracy_mse(valid_prediction.eval(), valid_output)))
    print('Test MSE: %.4f' % accuracy_mse(test_prediction.eval(), test_output))
    predicted_vs_actual = np.hstack((test_prediction.eval(), test_output))

print(np.corrcoef(predicted_vs_actual[:,0],predicted_vs_actual[:,2]))
print(np.corrcoef(predicted_vs_actual[:,1],predicted_vs_actual[:,3]))


