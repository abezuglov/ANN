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
import scipy
import math

#url = 'http://mrtee.europa.renci.org/~bblanton/ANN/'
#to = "../data"

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

#data_filename = maybe_download('ann_dataset1.tar', 5642240)

# Ten output data set
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
  
#data_filename = "../data/ann_dataset_10points.tar"
#data_files = maybe_extract(data_filename)

# Load files and produce dataset
def maybe_load(file_names):
    names = ('index','time', 'long', 'lat', 'param1', 'param2', 'param3', 'param4', 'out1', 'out2',
            'out3', 'out4','out5', 'out6','out7', 'out8','out9', 'out10',)
    datafile_length = 193
    dataset = np.ndarray(shape=(len(file_names), datafile_length, len(names)))
    for i in range(0,len(file_names)):
        a = np.loadtxt(file_names[i])
        a = np.asarray([x for xs in a for x in xs],dtype='d').reshape([datafile_length,len(names)])
        dataset[i,:,:] = a
        if i%100 == 0:
            print("Processed %d/%d \n"%(i,len(file_names)))
    return dataset

class BatchGenerator(object):
    def __init__(self, 
		data, 
		outs_index, 
		batch_size, 
		num_unrollings,
		input_moments = None,
		normalize_data = True):
        """
        Creates a batch generator
        data -- the dataset
        outs_index -- index of the first outputs component
        batch_size -- how many samples in each batch. Note the samples are NOT sequential in time!
        num_unrollings -- how many batches to return. The batches are sequential in time
        """
        self._data = data # the complete dataset
        self._data_size = data.shape[0] # how many samples in the dataset
        self._data_width = data.shape[1] # how many components in both inputs and outputs
        self._outs_index = outs_index # where the outputs start
        self._batch_size = batch_size
	if input_moments is None:
		self._input_moments = self.calc_input_moments()
	else:
		self._input_moments = input_moments
	if normalize_data:
		self._data[:outs_index] = (self._data[:outs_index] - self._input_moments[0])/self._input_moments[1]

        self._num_unrollings = num_unrollings
        segment = self._data_size // self._batch_size 
        self._cursor = [offset * segment for offset in range(self._batch_size)] # starting points for each batch
        self._last_batch = self._next_batch() # generate and save the first batch

    def calc_input_moments(self):
        """
        Calculate and return moments (mean, std)
        """
        means = [np.mean(self._data[:,i]) for i in range(self._outs_index)]
        stds = [np.std(self._data[:,i]) for i in range(self._outs_index)]
        return (means, stds)

    @property
    def input_moments(self):
        return self._input_moments
  
    def _next_batch(self):
        """
        Generate a single batch from the current cursor position in the data.
        Returns a tuple (inputs_batch,outputs_batch)
        """
        batch = np.zeros(shape=(self._batch_size, self._data_width), dtype = np.float) # prepare the batch array
        for b in range(self._batch_size): # cursors are indices where each data sample in the batch starts
            batch[b] = self._data[self._cursor[b],:] # copy the data
            self._cursor[b] = (self._cursor[b] + 1) % (self._data_size)
        return (batch[:,:self._outs_index],batch[:,self._outs_index:])
  
    def next(self):
        """
        Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings-1 new ones.
        """
        # make sure that the cursors stay within range
        self._cursor = [c%(self._data_size-self._num_unrollings) for c in self._cursor] 
        
        batches = [self._last_batch] # use the last batch as the first in the list
        for step in range(self._num_unrollings-1): # we only need _num_unrollings-1 new batches
            batches.append(self._next_batch())
        self._last_batch = batches[-1] # save the last batch to be reused next time
        return batches

def accuracy_mse(predictions, outputs):
    err = predictions-outputs
    return np.mean(err*err)

def Normalize(x, means, stds):
    return (x-means)/stds

def read_data_sets(directory = "../data/ann_dataset_10points", num_unrollings = 5, batch_size = 10):
	file_names = [os.path.join(directory, d) for d in sorted(os.listdir(directory)) if os.path.isdir(directory)]
	dataset = maybe_load(file_names)
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
	valid_start = 48 #int(test_percent/100.0*dataset.shape[0])
	train_start = 48 + 48 #int((test_percent+valid_percent)/100.0*dataset.shape[0])

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

	# Reshape the data and normalize
	train_dataset2 = train_dataset[:,:,1:7].reshape((-1, 6)).astype(np.float32)
	train_output = train_dataset[:,:,8:18].reshape((-1, 10)).astype(np.float32)

	# calculate means and stds for training dataset
	input_means = [np.mean(train_dataset2[:,i]) for i in range(train_dataset2.shape[1])]
	#print("Means: ",input_means)
	input_stds = [np.std(train_dataset2[:,i]) for i in range(train_dataset2.shape[1])]
	#print("STDs: ",input_stds)

	train_dataset2 = Normalize(train_dataset2, input_means, input_stds)

	valid_dataset2 = Normalize(valid_dataset[:,:,1:7].reshape((-1, 6)).astype(np.float32), input_means, input_stds)
	valid_output = valid_dataset[:,:,8:18].reshape((-1, 10)).astype(np.float32)

	test_dataset2 = Normalize(test_dataset[:,:,1:7].reshape((-1, 6)).astype(np.float32),input_means, input_stds)
	test_output = test_dataset[:,:,8:18].reshape((-1, 10)).astype(np.float32)
    
	train_batches = BatchGenerator(data = np.column_stack((train_dataset2,train_output)), 
					outs_index = train_dataset2.shape[1],
					batch_size = batch_size, 
					num_unrollings = num_unrollings,
					input_moments = (input_means, input_stds),
					normalize_data = False)

	del(train_dataset2)
	del(train_output)
	return train_batches, (valid_dataset2,valid_output), (test_dataset2,test_output)

def main(argv):
	read_data_sets()

