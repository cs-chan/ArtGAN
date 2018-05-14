import tensorflow as tf


def label_smoothing(labels, smooth):
    return labels * smooth + (1 - labels) * (1 - smooth) / (labels.get_shape().as_list()[1] - 1)


def log_sum_exp(x, axis=1):
    m = tf.reduce_max(x, axis=axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m), axis=axis))
