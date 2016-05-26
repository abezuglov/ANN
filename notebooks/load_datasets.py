from __future__ import print_function
import os
import sys
import tarfile
import numpy as np
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

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
#print(dataset.shape)

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

# Reshape the data and normalize

def normalize(x, means, stds):
    return (x-means)/stds

train_dataset2 = train_dataset[:,:,1:7].reshape((-1, 6)).astype(np.float32)
train_output = train_dataset[:,:,8:10].reshape((-1, 2)).astype(np.float32)

# calculate means and stds for training dataset
input_means = [np.mean(train_dataset2[:,i]) for i in range(train_dataset2.shape[1])]
input_stds = [np.std(train_dataset2[:,i]) for i in range(train_dataset2.shape[1])]
output_means = [np.mean(train_output[:,i]) for i in range(train_output.shape[1])]
output_stds = [np.std(train_output[:,i]) for i in range(train_output.shape[1])]

train_dataset2 = normalize(train_dataset2, input_means, input_stds)
train_output = normalize(train_output, output_means, output_stds)

#print(train_dataset2.shape)
#print(train_output)

valid_dataset2 = normalize(valid_dataset[:,:,1:7].reshape((-1, 6)).astype(np.float32), input_means, input_stds)
valid_output = normalize(valid_dataset[:,:,8:10].reshape((-1, 2)).astype(np.float32), output_means, output_stds)

test_dataset2 = normalize(test_dataset[:,:,1:7].reshape((-1, 6)).astype(np.float32),input_means, input_stds)
test_output = normalize(test_dataset[:,:,8:10].reshape((-1, 2)).astype(np.float32), output_means, output_stds)
