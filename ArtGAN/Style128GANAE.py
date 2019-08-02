import tensorflow as tf
from nn.layers import conv2d, linear, flatten, nnupsampling, batchnorm, gaussnoise, pool
from nn.activations import lrelu
from op import log_sum_exp
from data_loader import train_loader, validation_loader
from neon.backends import gen_backend
import numpy as np
from utils import drawblock, createfolders, OneHot, image_reshape
from imageio import imsave
import os


# Create folders to store images
gen_dir, real_dir, gen_dir128 = createfolders("./genimgs/Style128GANAE", "/gen", "/real", "/gen128")
# Create folder to store models
dir_name = './models/Style128GANAE'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

# Parameters
init_iter, max_iter = 0, 50000
display_iter = 100
eval_iter = 100
store_img_iter = 100
save_iter = 1000

lr_init, lrg_init = 0.001, 0.0001
batch_size = 100
zdim = 100
n_classes = 27
dropout = 0.2
im_size = [64, 64]
dname, gname = 'd_', 'g_'
tf.set_random_seed(1234)

# DataLoader
be = gen_backend(backend='cpu', batch_size=batch_size, datatype=np.float32)
root_files = './dataset/wikiart'
manifestfile = os.path.join(root_files, 'style-train-index.csv')
testmanifest = os.path.join(root_files, 'style-val-index.csv')
train = train_loader(manifestfile, root_files, be, h=im_size[0], w=im_size[1])
test = validation_loader(testmanifest, root_files, be, h=im_size[0], w=im_size[1], ncls=n_classes)
OneHot = OneHot(be, n_classes)

# Graph input
is_train = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)
x_n = tf.placeholder(tf.float32, [batch_size, 3, im_size[0], im_size[1]])
y = tf.placeholder(tf.float32, [batch_size, n_classes])
lr_tf = tf.placeholder(tf.float32)
z = tf.random_uniform([batch_size, zdim], -1, 1)
iny = tf.constant(np.tile(np.eye(n_classes, dtype=np.float32), [batch_size / n_classes + 1, 1])[:batch_size, :])


# Discriminator
def discriminator(inp, reuse=False):
    with tf.variable_scope('Encoder', reuse=reuse):
        # 64
        inp = gaussnoise(inp, std=0.05)
        conv1 = conv2d(inp, 128, kernel=3, strides=2, name=dname + 'conv1')
        conv1 = lrelu(conv1, 0.2)
        # 32
        conv2 = tf.nn.dropout(conv1, keep_prob)
        conv2 = conv2d(conv2, 256, kernel=3, strides=2, name=dname + 'conv2')
        conv2 = batchnorm(conv2, is_training=is_train, name=dname + 'bn2')
        conv2 = lrelu(conv2, 0.2)
        # 16
        conv3 = tf.nn.dropout(conv2, keep_prob)
        conv3 = conv2d(conv3, 512, kernel=3, strides=2, name=dname + 'conv3')
        conv3 = batchnorm(conv3, is_training=is_train, name=dname + 'bn3')
        conv3 = lrelu(conv3, 0.2)
        # 8
        conv3b = conv2d(conv3, 512, kernel=3, strides=1, name=dname + 'conv3b')
        conv3b = batchnorm(conv3b, is_training=is_train, name=dname + 'bn3b')
        conv3b = lrelu(conv3b, 0.2)

        conv4 = tf.nn.dropout(conv3b, keep_prob)
        conv4 = conv2d(conv4, 1024, kernel=3, strides=2, name=dname + 'conv4')
        conv4 = batchnorm(conv4, is_training=is_train, name=dname + 'bn4')
        conv4 = lrelu(conv4, 0.2)
        # 4
        flat = flatten(conv4)
        # Classifier
        clspred = linear(flat, n_classes, name=dname + 'cpred')
        # Decoder
        g1 = conv2d(conv4, nout=512, kernel=3, name=dname + 'deconv1')
        g1 = batchnorm(g1, is_training=tf.constant(True), name=dname + 'bn1g')
        g1 = lrelu(g1, 0.2)

        g2 = nnupsampling(g1, [8, 8])
        g2 = conv2d(g2, nout=256, kernel=3, name=dname + 'deconv2')
        g2 = batchnorm(g2, is_training=tf.constant(True), name=dname + 'bn2g')
        g2 = lrelu(g2, 0.2)

        g3 = nnupsampling(g2, [16, 16])
        g3 = conv2d(g3, nout=128, kernel=3, name=dname + 'deconv3')
        g3 = batchnorm(g3, is_training=tf.constant(True), name=dname + 'bn3g')
        g3 = lrelu(g3, 0.2)

        g4 = nnupsampling(g3, [32, 32])
        g4 = conv2d(g4, nout=64, kernel=3, name=dname + 'deconv4')
        g4 = batchnorm(g4, is_training=tf.constant(True), name=dname + 'bn4g')
        g4 = lrelu(g4, 0.2)

        g5 = nnupsampling(g4, [64, 64])
        g5 = conv2d(g5, nout=32, kernel=3, name=dname + 'deconv5')
        g5 = batchnorm(g5, is_training=tf.constant(True), name=dname + 'bn5g')
        g5 = lrelu(g5, 0.2)

        g5b = conv2d(g5, nout=3, kernel=3, name=dname + 'deconv5b')
        g5b = tf.nn.tanh(g5b)
        return clspred, g5b, flat


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
Opred_n, recon_n, _ = discriminator(x_n)
samples, samples128 = generator(z, iny)
Opred_g, recon_g, embed = discriminator(samples, reuse=True)

# Get trainable variables and split
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if dname in var.name]
g_vars = [var for var in t_vars if gname in var.name]
print [var.name for var in d_vars]
print [var.name for var in g_vars]

# Define D loss
lreal = log_sum_exp(Opred_n)
lfake = log_sum_exp(Opred_g)
cost_On = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Opred_n, labels=y))
cost_Dn = - tf.reduce_mean(lreal) + tf.reduce_mean(tf.nn.softplus(lreal))
cost_Dg_fake = tf.reduce_mean(tf.nn.softplus(lfake))
cost_msen = tf.reduce_mean(tf.square(recon_n - x_n)) * 0.5
cost_mseg = tf.reduce_mean(tf.square(recon_g - samples)) * 0.5
D_loss = cost_On + cost_Dn + cost_Dg_fake + cost_msen
# Define G loss
cost_Dg = - tf.reduce_mean(lfake) + tf.reduce_mean(tf.nn.softplus(lfake))
cost_Og = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Opred_g, labels=iny))
G_loss = cost_Dg + cost_Og + cost_mseg  # + pullaway_loss(embed)*0.1

# Define optimizer
d_optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_tf).minimize(D_loss, var_list=d_vars)
g_optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_tf).minimize(G_loss, var_list=g_vars)

# Evaluate model
Oaccuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Opred_n, 1), tf.argmax(y, 1)), tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()
# Reset train dataset
train.reset()
# Config for session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# Train
with tf.Session(config=config) as sess:
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=None)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i_iter in range(init_iter, max_iter):
        # Control lr
        if i_iter < 30000:
            lr = lr_init
            lrg = lrg_init
        else:
            lr = lr_init / 10.
            lrg = lrg_init / 10.
        # Fetch minibatch
        batch_x, batch_y = train.next()
        batch_x = image_reshape(batch_x.get(), im_size, input_format='tanh')
        batch_y = OneHot.transform(batch_y).get().transpose()
        # update discriminator
        _, lossDn, lossOn, lossFake = sess.run([d_optimizer, cost_Dn, cost_On, cost_Dg_fake], feed_dict={
            x_n: batch_x, y: batch_y,
            keep_prob: 1. - dropout, is_train: True, lr_tf: lr
        })
        # update generator
        _, gen_img, gen_img128 = sess.run([g_optimizer, samples, samples128], feed_dict={
            keep_prob: 1., is_train: True, lr_tf: lrg
        })
        # print losses
        if i_iter % display_iter == 0 or i_iter == max_iter - 1:
            print 'Iteration: %i, lossDn: %.2f, lossOn: %.2f, lossFake: %.2f' % (i_iter, lossDn, lossOn, lossFake)
        # Evaluate classification accuracy
        if i_iter % eval_iter == 0 or i_iter == max_iter - 1:
            total_Oaccuracy = 0.
            test.reset()
            for mb_idx, (batch_x, batch_y) in enumerate(test):
                batch_x = image_reshape(batch_x.get(), im_size, input_format='tanh')
                batch_y = batch_y.get().transpose()
                total_Oaccuracy += sess.run(Oaccuracy,
                                            feed_dict={x_n: batch_x, y: batch_y, keep_prob: 1., is_train: False})
            print 'Iteration %i, Accuracy: %.2f' % (i_iter, total_Oaccuracy / mb_idx)
        # Store images
        if i_iter % store_img_iter == 0 or i_iter == max_iter - 1:
            # Store Generated
            genmix_imgs = (np.transpose(gen_img, [0, 2, 3, 1]) + 1.) * 127.5
            genmix_imgs = np.uint8(genmix_imgs[:, :, :, ::-1])
            genmix_imgs = drawblock(genmix_imgs, n_classes, flip=True)
            imsave(os.path.join(gen_dir, '%i.jpg' % i_iter), genmix_imgs)
            # Store Generated 96
            genmix_imgs = (np.transpose(gen_img128, [0, 2, 3, 1]) + 1.) * 127.5
            genmix_imgs = np.uint8(genmix_imgs[:, :, :, ::-1])
            genmix_imgs = drawblock(genmix_imgs, n_classes, flip=True)
            imsave(os.path.join(gen_dir128, '%i.jpg' % i_iter), genmix_imgs)
            # Store Real
            real_imgs = (np.transpose(batch_x, [0, 2, 3, 1]) + 1.) * 127.5
            real_imgs = np.uint8(real_imgs[:, :, :, ::-1])
            real_imgs = drawblock(real_imgs, 10)
            imsave(os.path.join(real_dir, '%i.jpg' % i_iter), real_imgs)
        # Store model
        if i_iter % save_iter == 0 or i_iter == max_iter - 1 or i_iter == max_iter:
            save_path = saver.save(sess, dir_name + '/cdgan%i.ckpt' % i_iter)
    coord.request_stop()
    coord.join(threads)
