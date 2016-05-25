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

url = 'http://mrtee.europa.renci.org/~bblanton/ANN/'
to = "../data"

def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    print(os.path.join(to,filename))
    print(url+filename)
    if force or not os.path.exists(os.path.join(to,filename)):
        filename, _ = urlretrieve(url + filename, os.path.join(to,filename))
    statinfo = os.stat(os.path.join(to,filename))
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
          'Failed to verify' + filename + '. Can you get to it with a browser?')
    return filename

data_filename = maybe_download('ann_dataset1.tar', 5642240)

# Extract files from the archive
def maybe_extract(filename, force=False):
    extract_folder = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    root = os.path.dirname(filename)
    if os.path.isdir(extract_folder) and not force:
    # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(path = root)
        tar.close()
    data_files = [
        os.path.join(extract_folder, d) for d in sorted(os.listdir(extract_folder))
        if os.path.isdir(extract_folder)]
    return data_files
  
data_filename = "../data/ann_dataset1.tar"
data_files = maybe_extract(data_filename)

# Load files and produce dataset
def maybe_load(file_names):
    names = ('index','time', 'long', 'lat', 'param1', 'param2', 'param3', 'param4', 'out1', 'out2')
    datafile_length = 193
    dataset = np.ndarray(shape=(len(file_names), datafile_length, len(names)))
    for i in range(0,len(file_names)):
        a = np.loadtxt(file_names[i])
        a = np.asarray([x for xs in a for x in xs],dtype='d').reshape([datafile_length,len(names)])
        dataset[i,:,:] = a
        if i%100 == 0:
            print("Processed %d/%d \n"%(i,len(file_names)))
    return dataset

dataset = maybe_load(data_files)
print(dataset.shape)

# train, validation, and test dataset percentages
train_percent = 70
valid_percent = 15
test_percent = 15

# train, validation, and test dataset indices
# test: test_start : valid_start-1
# validation: valid_start : train_start-1
# training: train_start : dataset.shape[0]
test_start = 0 
valid_start = int(test_percent/100.0*dataset.shape[0])
train_start = int((test_percent+valid_percent)/100.0*dataset.shape[0])

# Shuffle file indices
file_indices = range(dataset.shape[0])
np.random.shuffle(file_indices)

# Assign datasets
test_dataset = np.array([dataset[j,:,:] for j in [file_indices[i] for i in range(test_start, valid_start)]])
valid_dataset = np.array([dataset[j,:,:] for j in [file_indices[i] for i in range(valid_start, train_start)]])
train_dataset = np.array([dataset[j,:,:] for j in [file_indices[i] for i in range(train_start, dataset.shape[0])]])

# Save memory
del(dataset)

print("Test dataset: "+str(test_dataset.shape))
print("Validation dataset: "+str(valid_dataset.shape))
print("Training dataset: "+str(train_dataset.shape))

def accuracy_mse(predictions, outputs):
    err = predictions-outputs
    return np.mean(err*err)

def Normalize(x, means, stds):
    return (x-means)/stds

# Reshape the data and normalize

train_dataset2 = train_dataset[:,:,1:7].reshape((-1, 6)).astype(np.float32)
train_output = train_dataset[:,:,8:10].reshape((-1, 2)).astype(np.float32)

# calculate means and stds for training dataset
input_means = [np.mean(train_dataset2[:,i]) for i in range(train_dataset2.shape[1])]
input_stds = [np.std(train_dataset2[:,i]) for i in range(train_dataset2.shape[1])]
output_means = [np.mean(train_output[:,i]) for i in range(train_output.shape[1])]
output_stds = [np.std(train_output[:,i]) for i in range(train_output.shape[1])]

train_dataset2 = Normalize(train_dataset2, input_means, input_stds)
train_output = Normalize(train_output, output_means, output_stds)

print(train_dataset2.shape)
print(train_output)

valid_dataset2 = Normalize(valid_dataset[:,:,1:7].reshape((-1, 6)).astype(np.float32), input_means, input_stds)
valid_output = Normalize(valid_dataset[:,:,8:10].reshape((-1, 2)).astype(np.float32), output_means, output_stds)

test_dataset2 = Normalize(test_dataset[:,:,1:7].reshape((-1, 6)).astype(np.float32),input_means, input_stds)
test_output = Normalize(test_dataset[:,:,8:10].reshape((-1, 2)).astype(np.float32), output_means, output_stds)


# Deep ANN
batch_size = 20*193
hidden_nodes_1 = 40
hidden_nodes_2 = 25
hidden_nodes_3 = 10

num_steps = 500001
starter_learning_rate = 0.05
rate_decay = 0.1

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


