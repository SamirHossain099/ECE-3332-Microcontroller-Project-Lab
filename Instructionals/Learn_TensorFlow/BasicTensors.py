import tensorflow as tf
t = tf.zeros([4,3,2,1])
box = tf.ones([2,5])
t = tf.reshape(t, [2, -1])
print(t)