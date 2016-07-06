from __future__ import print_function
import os
import sys
import tarfile
import numpy as np
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

def maybe_download(url_from, to, filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    print(os.path.join(to,filename))
    print(url_from+filename)
    if force or not os.path.exists(os.path.join(to,filename)):
        filename, _ = urlretrieve(url_from + filename, os.path.join(to,filename))
    statinfo = os.stat(os.path.join(to,filename))
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
          'Failed to verify' + filename + '. Can you get to it with a browser?')
    return filename

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
    file_names = [
        os.path.join(extract_folder, d) for d in sorted(os.listdir(extract_folder))
        if os.path.isdir(extract_folder)]
    return file_names
  
# Load files and produce dataset
def maybe_load(file_names):
    names = ('index','time', 'long', 'lat', 'param1', 'param2', 'param3', 'param4', 
             'out1', 'out2', 'out3', 'out4', 'out5', 'out6', 'out7', 'out8', 'out9', 'out10')
    datafile_length = 193
    dataset = np.ndarray(shape=(len(file_names), datafile_length, len(names)))
    for i in range(0,len(file_names)):
        a = np.loadtxt(file_names[i])
        a = np.asarray([x for xs in a for x in xs],dtype='d').reshape([datafile_length,len(names)])
        dataset[i,:,:] = a
        if i%100 == 0:
            print("Processed %d/%d \n"%(i,len(file_names)))
    return dataset

# Class representing datasets
class Dataset(object):
    """
    The dataset is responsible for calculating moments and delivering data in patches
    the input data is normalized
    """
    def __init__(self,
                 inputs,
                 outputs,
                 means = None,
                 stds = None,
                 normalize_data = True):
        """
        Class initialization
        inputs, outputs -- ndarrays with input and output data
        means, stds -- provided means and std's for input data
        normalize_inputs -- the input data will be normalized in the constructor
        """
        self._inputs = inputs
        self._outputs = outputs
        #self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = inputs.shape[0]

        if means is None:
            print("calculate new means, stds for dataset with %d samples"%inputs.shape[0])
            self._means, self._stds = self.calc_moments()
        else:
            print("using provided means, stds for dataset with %d samples"%inputs.shape[0])
            self._means = means
            self._stds = stds
	if normalize_data:
		self._inputs = (self._inputs - self._means)/self._stds


    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def num_examples(self):
        return self._num_examples

    #@property
    #def epochs_completed(self):
    #    return self._epochs_completed

    @property
    def means(self):
        return self._means

    @property
    def stds(self):
        return self._stds

    @property
    def num_hurricanes(self):
        return self._num_examples//193

    def calc_moments(self):
        """
        Calculate and return moments (mean, std)
        """
        means = [np.mean(self.inputs[:,i]) for i in range(self.inputs.shape[1])]
        stds = [np.std(self.inputs[:,i]) for i in range(self.inputs.shape[1])]
        return means, stds

    def next_batch(self, batch_size = 0):
        """
        Returns the next batch of data of size batch_size. If batch_size is not specified or larger than the dataset,
        the whole dataset is returned. 
	When the dataset is not divisible into batches, the last batch will be smaller
        """
        if batch_size > 0 and batch_size <= self._num_examples:
	    start = self._index_in_epoch
	    end = start + batch_size
	    self._index_in_epoch = end
	    if end >= self._num_examples:
		# technically, if end == _num_examples, it is still a "valid" data sample consisting
		# of a single record. However, it is ignored
		end = self._num_examples
		self._index_in_epoch = 0
            return self._inputs[start:end], self.outputs[start:end]
        else:
            return self._inputs, self.outputs

    def get_full(self):
        """
        Returns complete dataset
        """
        return self._inputs, self.outputs

#def read_data_sets(url_from = 'http://mrtee.europa.renci.org/~bblanton/ANN/', 
#                   to = "../data", 
#                   file_name = 'ann_dataset1.tar',
#                   file_size = 5642240):
#    data_filename = maybe_download(url_from, to, file_name, file_size)
#    #data_filename = "../data/ann_dataset1.tar"
#    file_names = maybe_extract(os.path.join(to,file_name))
#    #load full dataset from files

def read_data_sets(directory = "../data/ann_dataset_10points"):
    file_names = [
        os.path.join(directory, d) for d in sorted(os.listdir(directory))
        if os.path.isdir(directory)]

    ds = maybe_load(file_names)
    
    # train, validation, and test dataset percentages
    # Not currently used. The start/stop indices are hard-coded
    #train_percent = 70
    #valid_percent = 15
    #test_percent = 15

    # train, validation, and test dataset indices
    # test: test_start : valid_start-1
    # validation: valid_start : train_start-1
    # training: train_start : dataset.shape[0]
    #test_start = 0 
    #valid_start = int(test_percent/100.0*ds.shape[0])
    #train_start = int((test_percent+valid_percent)/100.0*ds.shape[0])

    
    # Option 1: Training dataset contains 228 samples (batches: 19,57,114, and 228)
    # test: 48, validation: 48, training: 228
    test_start = 0
    valid_start = 48
    train_start = 48+48
    """
    # Option 2: Training dataset contains 225 samples (batches: 3,5,9,15,45, and 225)
    # test: 50, validation: 49, training: 225
    test_start = 0
    valid_start = 50
    train_start = 50+49
    """

    # Shuffle file indices
    file_indices = range(ds.shape[0])
    np.random.shuffle(file_indices)

    # Assign datasets
    test_dataset = np.array([ds[j,:,:] for j in [file_indices[i] for i in range(test_start, valid_start)]])
    valid_dataset = np.array([ds[j,:,:] for j in [file_indices[i] for i in range(valid_start, train_start)]])
    train_dataset = np.array([ds[j,:,:] for j in [file_indices[i] for i in range(train_start, ds.shape[0])]])

    """
    The input files are formatted as:
    0      1    2    3    4    5    6    7    8     9         n
    seq_no inp1 inp2 inp3 inp4 inp5 inp6 inp7 outp1 outp2 ... outp_n

    Valid inputs are: 1:7 (inp7 is excluded as its std equals to 0), a total of 6 inputs
    Valid outputs are: 8:n 
    """
    first_outs_index = 8 # First index of the outputs
    max_outs_index = ds.shape[2] + 1 # Find the last index of the outputs
    outputs_num = max_outs_index - 1 - first_outs_index # Number of outputs for reshaping
    first_inps_index = 1
    max_inps_index = 7
    inputs_num = max_inps_index - first_inps_index

    # Save memory
    del(ds)

    # skip the first column (seq. no), reshape
    train_dataset2 = train_dataset[:,:,first_inps_index:max_inps_index].reshape((-1, inputs_num)).astype(np.float32)
    train_output = train_dataset[:,:,first_outs_index:max_outs_index].reshape((-1, outputs_num)).astype(np.float32)

    # calculate means and stds for training dataset
    input_means = [np.mean(train_dataset2[:,i]) for i in range(train_dataset2.shape[1])]
    input_stds = [np.std(train_dataset2[:,i]) for i in range(train_dataset2.shape[1])]

    valid_dataset2 = valid_dataset[:,:,first_inps_index:max_inps_index].reshape((-1, inputs_num)).astype(np.float32)
    valid_output = valid_dataset[:,:,first_outs_index:max_outs_index].reshape((-1, outputs_num)).astype(np.float32)

    test_dataset2 = test_dataset[:,:,first_inps_index:max_inps_index].reshape((-1, inputs_num)).astype(np.float32)
    test_output = test_dataset[:,:,first_outs_index:max_outs_index].reshape((-1, outputs_num)).astype(np.float32)

    train = Dataset(train_dataset2, train_output)

    # Create validation and test datasets. The moments are added for compatibility.
    # Only training dataset moments are used.
    valid = Dataset(valid_dataset2, valid_output, means = train.means, stds = train.stds)
    test = Dataset(test_dataset2, test_output, means = train.means, stds = train.stds)

    del(train_dataset2)
    del(train_output)
    del(valid_dataset2)
    del(valid_output)
    del(test_dataset2)
    del(test_output)
    print("Num hurricanes in train %d, validation %d, test %d" % (train.num_hurricanes, valid.num_hurricanes, test.num_hurricanes))
    return train, valid, test


