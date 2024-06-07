"""
Tensors:
    A tensor is a generalization of vectors and matrices to potentially higher dimensions.
    Each tensor represents a partially defined computation that will eventually produce a value. 
    TensorFlow programs work by building a graph of tensor objects that detail how tensors are related. 
    Each tensor has a data type and a shape.
    Data types include: float32, int32, string, and others
    Shape: represents the dimension of data 
"""
import tensorflow as tf

# Creating tensors:

string = tf.Variable("This is a string",tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

# Rank/Degree of tensors:
"""
Rank 1 is just a 1D array
Rank 2 tensor is a 2D array
"""
rank1_tensor = tf.Variable(["Test"], tf.string) 
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)

#to determine the rank of a tensor, it is the deepest level of a nest list
#This can also be determined using tf.rank(<tensor>)
tf.rank(rank2_tensor)

# tensor shape tells us how many elements are in each dimension
rank2_tensor.shape
