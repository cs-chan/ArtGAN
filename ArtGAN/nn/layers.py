import tensorflow as tf
import numpy as np


def instance_norm(net, train=False, epsilon=1e-8, name='in'):
    """use shape NCHW"""
    with tf.variable_scope(name):
        mu, sigma_sq = tf.nn.moments(net, [2, 3], keep_dims=True)
        normalized = (net - mu) / tf.sqrt(sigma_sq + epsilon)
        if train:
            var_shape = net.get_shape().as_list()[1]
            shift = tf.get_variable('shift', shape=[1, var_shape, 1, 1], initializer=tf.constant_initializer(0.))
            scale = tf.get_variable('scale', shape=[1, var_shape, 1, 1], initializer=tf.constant_initializer(1.))
            normalized = scale * normalized + shift
        return normalized


def adain(net1, net2, epsilon=1e-9, name='in'):
    """use shape NCHW"""
    with tf.variable_scope(name):
        mu, sigma_sq = tf.nn.moments(net1, [2, 3], keep_dims=True)
        normalized = (net1 - mu) / tf.sqrt(sigma_sq + epsilon)

        shift, scale = tf.nn.moments(net2, [2, 3], keep_dims=True)
        normalized = tf.sqrt(scale + epsilon) * normalized + shift
        return normalized


def adain2(net1, shift, scale, epsilon=1e-9, name='in'):
    """use shape NCHW"""
    with tf.variable_scope(name):
        mu, sigma_sq = tf.nn.moments(net1, [2, 3], keep_dims=True)
        normalized = (net1 - mu) / tf.sqrt(sigma_sq + epsilon)
        normalized = tf.sqrt(scale + epsilon) * normalized + shift
        return normalized


def layernorm(x, epsilon=1e-5, name='lnconv'):
    """Layer Normalization for conv. x must be [NCHW]"""
    shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        beta = tf.get_variable("beta", [1, shape[1], 1, 1], initializer=tf.constant_initializer(0.))
        gamma = tf.get_variable("gamma", [1, shape[1], 1, 1], initializer=tf.constant_initializer(1.))
        mean, var = tf.nn.moments(x, range(1, len(shape)), keep_dims=True)
        return beta * (x - mean) / tf.sqrt(var + epsilon) + gamma


def batchnorm(inputT, is_training=tf.constant(True), name='bn'):
    # Note: is_training is tf.placeholder(tf.bool) type
    with tf.variable_scope(name) as vs:
        return tf.cond(is_training,
                       lambda: tf.contrib.layers.batch_norm(inputT, is_training=True, decay=0.9, fused=True,
                                                            updates_collections=None, center=True, scale=True,
                                                            scope=vs, data_format='NCHW'),
                       lambda: tf.contrib.layers.batch_norm(inputT, is_training=False, decay=0.9, fused=True,
                                                            updates_collections=None, center=True, scale=True,
                                                            scope=vs, data_format='NCHW', reuse=True))


def batchnorm_true(inputT, name='bn', fused=True):
    with tf.variable_scope(name) as vs:
        return tf.contrib.layers.batch_norm(inputT, is_training=True, decay=0.9, fused=fused,
                                            updates_collections=None, center=True, scale=True,
                                            scope=vs, data_format='NCHW')


def featurenorm(x, epsilon=1e-8, name='fn'):
    """
    Pixelwise feature vector normalization: PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION
    Use "NCHW". Works same for FC layers.
    """
    with tf.variable_scope(name):
        norm = tf.sqrt(tf.reduce_mean(tf.square(x), axis=1, keep_dims=True) + epsilon)
        return x / norm


def minibatch_std(x, name='ministd'):
    shapes = x.get_shape().as_list()
    with tf.variable_scope(name):
        _, var = tf.nn.moments(x, [0], keep_dims=True)
        return tf.concat([x, tf.tile(tf.reduce_mean(var, keep_dims=True), [shapes[0], 1, shapes[2], shapes[3]])], axis=1)


def conv2d(x, nout, kernel=3, std=0.02, use_b=False, strides=1, name='conv2d', print_struct=True, pad='SAME'):
    if pad == 0:
        pad = 'VALID'
    with tf.variable_scope(name):
        W = tf.get_variable('W', [kernel, kernel, x.get_shape()[1], nout],
                            initializer=tf.random_normal_initializer(stddev=std))
        conv = tf.nn.conv2d(x, W, strides=[1, 1, strides, strides], padding=pad, data_format='NCHW')
        if print_struct:
            print conv.name + ': ' + str(conv.get_shape().as_list())
        if use_b:
            b = tf.get_variable('b', [nout], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, b, data_format='NCHW')
        return conv


def deconv2d(x, nout, kernel=3, std=0.02, use_b=False, strides=2, name='deconv2d'):
    # Not tested yet!
    with tf.variable_scope(name):
        shape = x.get_shape().as_list()
        W = tf.get_variable('W', [kernel, kernel, nout, shape[1]],
                            initializer=tf.random_normal_initializer(stddev=std))
        deconv = tf.nn.conv2d_transpose(x, W, [shape[0], nout, shape[2] * strides, shape[3] * strides],
                                        [1, 1, strides, strides], data_format='NCHW')
        if use_b:
            b = tf.get_variable('b', [nout], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(x, b, data_format='NCHW')
        print deconv.name + ': ' + str(deconv.get_shape().as_list())
        return deconv


def pool(x, fsize=2, strides=2, op='max', pad='SAME'):
    assert pad in ['VALID', 'SAME']
    if op is 'max':
        pooled = tf.nn.max_pool(x, ksize=[1, 1, fsize, fsize], strides=[1, 1, strides, strides], padding=pad,
                                data_format='NCHW')
    elif op is 'avg':
        pooled = tf.nn.avg_pool(x, ksize=[1, 1, fsize, fsize], strides=[1, 1, strides, strides], padding=pad,
                                data_format='NCHW')
    else:
        raise ValueError('op only supports max or avg!')
    print pooled.name + ': ' + str(pooled.get_shape().as_list())
    return pooled


def linear(x, nout, std=0.02, use_b=False, init_b=0.0, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable('W', [x.get_shape()[-1], nout], initializer=tf.random_normal_initializer(stddev=std))
        lout = tf.matmul(x, W)
        if use_b:
            b = tf.get_variable('b', [nout], initializer=tf.constant_initializer(init_b))
            lout = tf.nn.bias_add(lout, b)
        print lout.name + ': ' + str(lout.get_shape().as_list())
        return lout


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def nnupsampling(inp, size):
    upsample = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(inp, [0, 2, 3, 1]), size), [0, 3, 1, 2])
    print upsample.name + ': ' + str(upsample.get_shape().as_list())
    return upsample


def subpixel(inp, nfm, upscale=2, name='subpixel'):
    # assert inp.get_shape().as_list()[1] % upscale == 0
    output = conv2d(inp, nout=nfm * (upscale ** 2), kernel=1, name=name, print_struct=False)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, upscale)
    output = tf.transpose(output, [0, 3, 1, 2])
    print name + ': ' + str(output.get_shape().as_list())
    return output


def gaussnoise(x, std=0.2, wrt=False):
    if wrt:
        return x + x * tf.random_normal(tf.shape(x), stddev=std)
    else:
        return x + tf.random_normal(tf.shape(x), stddev=std)


def total_variation(preds):
    # total variation denoising
    b, c, w, h = preds.get_shape().as_list()
    y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :w - 1, :, :])
    x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :h - 1, :])
    tv_loss = 2 * (x_tv + y_tv) / b / w / h / c
    return tv_loss


def pullaway_loss(embeddings):
    """
    Pull Away loss calculation
    :param embeddings: The embeddings to be orthogonalized for varied faces. Shape [batch_size, embeddings_dim]
    :return: pull away term loss
    """
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    similarity = tf.matmul(
        normalized_embeddings, normalized_embeddings, transpose_b=True)
    similarity -= tf.diag(tf.diag_part(similarity))
    batch_size = tf.cast(tf.shape(embeddings)[0], tf.float32)
    pt_loss = tf.reduce_sum(similarity) / (batch_size * (batch_size - 1))
    return pt_loss


def minibatch_discrimination(inp, num_kernels=5, kernel_dim=3):
    x = linear(inp, num_kernels * kernel_dim)
    activation = tf.reshape(x, [-1, num_kernels, kernel_dim])
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([inp, minibatch_features], 1)


def gramatrix(x, scale=0.1):
    shape = x.get_shape().as_list()
    x_reshape = tf.reshape(x, [-1, shape[1], shape[2] * shape[3]])
    x_reshape = tf.matmul(x_reshape, x_reshape, transpose_b=True) * scale
    print 'gramatrix: ' + str(x_reshape.get_shape().as_list())
    return x_reshape


def _phase_shift(I, r):
    # Helper function with main phase shift operation
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(1, a, X)  # a, [bsize, b, r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
    X = tf.split(1, b, X)  # b, [bsize, a*r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  #
    # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a * r, b * r, 1))


def PS(X, r, scale=2):
    # Main OP that you can arbitrarily use in you tensorflow code
    Xc = tf.split(tf.transpose(X, [0, 2, 3, 1]), scale, 3)
    X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    X = tf.transpose(X, [0, 3, 1, 2])
    print 'PhaseShift: ' + str(X.get_shape().as_list())
    return X
