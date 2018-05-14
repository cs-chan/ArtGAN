import tensorflow as tf


def lrelu(x, alpha=0.1):
    return tf.maximum(alpha*x, x)
