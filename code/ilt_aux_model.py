import tensorflow as tf

def weight_variable(name, shape, std = 0.1):
    """
    Returns a shared TF weight variable with given shape. The weights are normally distributed with mean = 0, stddev = 0.1
    shape -- shape of the variable, i.e. [4,5] matrix of 4x5
    """
    with tf.device('/cpu:0'):
        initial = tf.truncated_normal_initializer(stddev = std)
        var = tf.get_variable(name, shape, initializer = initial)
    return var

def bias_variable(name, shape):
    """
    Returns a shared TF bias variable with given shape. The biases are initially at 0.1
    shape -- shape of the variable, i.e. [4] -- vector of length 4
    """
    with tf.device('/cpu:0'):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable(name, shape, initializer = initial)
    return var

def variable_summaries(var, name):
    """
    Add multiple (excessive) summaries (statistics) for a TF variable
    var -- TF variable
    name -- variable name
    """
    #mean = tf.reduce_mean(var)
    #tf.scalar_summary(name+'/mean', mean)
    #stddev = tf.reduce_mean(tf.reduce_sum(tf.square(var-mean)))
    #tf.scalar_summary(name+'/stddev', stddev)
    #_min = tf.reduce_min(var)
    #tf.scalar_summary(name+'/min', _min)
    #_max = tf.reduce_max(var)
    #tf.scalar_summary(name+'/max', _max)
    #tf.histogram_summary(name, var)
    #tf.scalar_summary(name+'/sparsity',tf.nn.zero_fraction(var))

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act = tf.tanh, std = 0.1):
    """
    Creates and returns NN layer
    input_tensor -- TF tensor at layer input
    input_dim -- size of layer input
    output_dim -- size of layer output
    layer_name -- name of the layer for summaries (statistics)
    act -- nonlinear activation function
    """
    with tf.variable_scope(layer_name):
        weights = weight_variable('weights',[input_dim, output_dim], std)
        #variable_summaries(weights, layer_name+'/weights')

        biases = bias_variable('biases',[output_dim])
        #variable_summaries(biases, layer_name+'/biases')

        preactivate = tf.matmul(input_tensor, weights)+biases
        #tf.histogram_summary(layer_name+'/pre_activations', preactivate)
        if act is not None:
            activations = act(preactivate, 'activation')
        else:
            activations = preactivate
        #tf.histogram_summary(layer_name+'/activations', activations)
    return activations
