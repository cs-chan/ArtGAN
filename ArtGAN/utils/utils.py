import numpy as np
import unicodedata
import os


class OneHot(object):
    def __init__(self, be, nclasses):
        self.be = be
        self.output = be.iobuf(nclasses, parallelism='Data')

    def transform(self, t):
        self.output[:] = self.be.onehot(t, axis=0)
        return self.output


def image_reshape(img, shape, dtype=np.float32, data_format='NCHW', input_format='sigmoid'):
    if data_format is 'NCHW':
        df = (3, 0, 1, 2)
    elif data_format is 'NHWC':
        df = (3, 1, 2, 0)
    else:
        raise ValueError('data_format only support NCHW or NHWC!')
    if input_format is 'tanh':
        img = img.astype(dtype).reshape((3, shape[0], shape[1], -1)) / 127.5 - 1.
    elif input_format is 'sigmoid':
        img = img.astype(dtype).reshape((3, shape[0], shape[1], -1)) / 255.
    else:
        raise ValueError('Unsupported input format!')
    img = np.transpose(img, df)
    return img


def drawblock(arr, num_class=10, fixed=False, flip=False, split=False):
    """
    draw images in block
    :param arr: array of images. format='NHWC'. sequence=[cls1,cls2,cls3,...,clsN,cls1,cls2,...clsN]
    :param num_class: number of class. default as number of images across height. Use flip=True to set number of width as across width instead
    :param fixed: force number of number of width == number of height
    :param flip: flip
    :param split: set an int to split. currently only support split horizontally
    :return: blocks of images
    """
    n_im = arr.shape[0]
    h_im = arr.shape[1]
    w_im = arr.shape[2]
    c_im = arr.shape[3]

    if flip:
        num_w = num_class
        num_h = np.ceil(np.float(n_im) / num_w) if not fixed else num_w
        if fixed and isinstance(fixed, int):
            num_h = fixed
    else:
        num_h = num_class
        num_w = np.ceil(np.float(n_im) / num_h) if not fixed else num_h
        if fixed and isinstance(fixed, int):
            num_w = fixed
    h_block = (h_im + 2) * num_h - 2
    w_block = (w_im + 2) * num_w - 2
    newarr = np.zeros((int(h_block), int(w_block), int(c_im)), dtype=np.uint8)
    for i in xrange(n_im):
        if i > num_w * num_h - 1:
            break
        if flip:
            wk = i % num_w
            hk = i // num_w
        else:
            hk = i % num_h
            wk = i // num_h
        wk = int(wk)
        hk = int(hk)
        newarr[hk*(h_im+2):hk*(h_im+2)+h_im, wk*(w_im+2):wk*(w_im+2)+w_im, :] = arr[i]

    if split:
        temp = newarr
        newnh = int(np.ceil(float(num_class) / split))
        newh = (h_im + 2) * newnh - 2
        neww = int(w_block * split + 2)
        newarr = np.zeros((newh, neww, int(c_im)), dtype=np.uint8)
        for i in range(split):
            if not num_class % split == 0 and i == split - 1:
                newarr[:-h_im-2, i * w_block+i*2:(i + 1) * w_block+(i+1)*2, :] = temp[i * newh+i*2:, :, :]
            else:
                newarr[:, i*w_block+i*2:(i+1)*w_block, :] = temp[i*newh+i*2:(i+1)*newh, :, :]
    return newarr


def readclasslabels(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        labels.append(line.split()[1])
    return labels


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def datasetweights(dataset):
    dataset.reset()
    weights = np.zeros(10)
    for x, t in dataset:
        t = t.get().argmax(axis=0)
        i, c = np.unique(t, return_counts=True)
        for ii, cc in zip(i, c):
            weights[ii] += cc
    weights /= dataset.ndata
    return weights


def specialchar(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def createfolders(par, *args):
    if not os.path.exists(par):
        os.mkdir(par)
    return_list = []
    for arg in args:
        new_dir = par + arg
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        return_list.append(new_dir)
    return return_list
