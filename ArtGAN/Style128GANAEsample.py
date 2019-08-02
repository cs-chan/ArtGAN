import tensorflow as tf
from nn.layers import conv2d, linear, nnupsampling, batchnorm, pool
from nn.activations import lrelu
import numpy as np
from utils import drawblock, createfolders
from imageio import imsave
import os


# Create folders to store images
gen_dir, gen_dir128 = createfolders("./genimgs/Style128GANAEsample", "/gen", "/gen128")

# Parameters
batch_size = 100
zdim = 100
n_classes = 27
im_size = [64, 64]
gname = 'g_'
tf.set_random_seed(777)  # use different seed to generate different set of images

# Graph input
z = tf.random_uniform([batch_size, zdim], -1, 1)
iny = tf.constant(np.tile(np.eye(n_classes, dtype=np.float32), [batch_size / n_classes + 1, 1])[:batch_size, :])


# Generator
def generator(inp_z, inp_y, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        inp = tf.concat([inp_z, inp_y], 1)
        sz = 4
        g1 = linear(inp, 512 * sz * sz, name=gname + 'deconv1')
        g1 = batchnorm(g1, is_training=tf.constant(True), name=gname + 'bn1g')
        g1 = lrelu(g1, 0.2)
        g1_reshaped = tf.reshape(g1, [-1, 512, sz, sz])
        print 'genreshape: ' + str(g1_reshaped.get_shape().as_list())

        g2 = nnupsampling(g1_reshaped, [8, 8])
        g2 = conv2d(g2, nout=512, kernel=3, name=gname + 'deconv2')
        g2 = batchnorm(g2, is_training=tf.constant(True), name=gname + 'bn2g')
        g2 = lrelu(g2, 0.2)

        g3 = nnupsampling(g2, [16, 16])
        g3 = conv2d(g3, nout=256, kernel=3, name=gname + 'deconv3')
        g3 = batchnorm(g3, is_training=tf.constant(True), name=gname + 'bn3g')
        g3 = lrelu(g3, 0.2)

        g4 = nnupsampling(g3, [32, 32])
        g4 = conv2d(g4, nout=128, kernel=3, name=gname + 'deconv4')
        g4 = batchnorm(g4, is_training=tf.constant(True), name=gname + 'bn4g')
        g4 = lrelu(g4, 0.2)

        g5 = nnupsampling(g4, [64, 64])
        g5 = conv2d(g5, nout=64, kernel=3, name=gname + 'deconv5')
        g5 = batchnorm(g5, is_training=tf.constant(True), name=gname + 'bn5g')
        g5 = lrelu(g5, 0.2)

        g5b = conv2d(g5, nout=64, kernel=3, name=gname + 'deconv5b')
        g5b = batchnorm(g5b, is_training=tf.constant(True), name=gname + 'bn5bg')
        g5b = lrelu(g5b, 0.2)

        g6 = nnupsampling(g5b, [128, 128])
        g6 = conv2d(g6, nout=32, kernel=3, name=gname + 'deconv6')
        g6 = batchnorm(g6, is_training=tf.constant(True), name=gname + 'bn6g')
        g6 = lrelu(g6, 0.2)

        g6b = conv2d(g6, nout=3, kernel=3, name=gname + 'deconv6b')
        g6b = tf.nn.tanh(g6b)
        g6b_64 = pool(g6b, fsize=3, strides=2, op='avg')
        return g6b_64, g6b

# Call functions
samples, samples128 = generator(z, iny)

# Initialize the variables
init = tf.global_variables_initializer()
# Config for session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# Generate
with tf.Session(config=config) as sess:
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=None)
    saver.restore(sess=sess, save_path='./models/Style128GANAE/cdgan49999.ckpt')

    # run generator
    gen_img, gen_img128 = sess.run([samples, samples128])

    # Store Generated
    genmix_imgs = (np.transpose(gen_img, [0, 2, 3, 1]) + 1.) * 127.5
    genmix_imgs = np.uint8(genmix_imgs[:, :, :, ::-1])
    genmix_imgs = drawblock(genmix_imgs, n_classes, fixed=3, flip=False)
    imsave(os.path.join(gen_dir, 'sample.jpg'), genmix_imgs)
    # Store Generated 128
    genmix_imgs = (np.transpose(gen_img128, [0, 2, 3, 1]) + 1.) * 127.5
    genmix_imgs = np.uint8(genmix_imgs[:, :, :, ::-1])
    genmix_imgs = drawblock(genmix_imgs, n_classes, fixed=3, flip=False)
    imsave(os.path.join(gen_dir128, 'sample.jpg'), genmix_imgs)
