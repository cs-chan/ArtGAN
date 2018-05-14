from configargparse import ArgParser
from PIL import Image
import logging
import numpy as np
import os


def transform_and_save(img_arr, output_filename):
    """
    Takes an image and optionally transforms it and then writes it out to output_filename
    """
    img = Image.fromarray(img_arr)
    img.save(output_filename)


class Ingest(object):
    def __init__(self, input_dir, out_dir, target_size=96, skipimg=False):
        np.random.seed(0)
        self.skipimg = skipimg
        self.out_dir = out_dir
        self.input_dir = input_dir

        self.manifests = dict()
        for setn in ('train', 'val'):
            self.manifests[setn] = os.path.join(self.out_dir, '{}-index.csv'.format(setn))

        self.target_size = target_size
        self.trainpairlist = {}
        self.valpairlist = {}
        self.labels = range(10)

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        self.outimgdir = os.path.join(self.out_dir, 'images')
        if not os.path.exists(self.outimgdir):
            os.mkdir(self.outimgdir)
            os.mkdir(os.path.join(self.outimgdir, 'train'))
            os.mkdir(os.path.join(self.outimgdir, 'val'))
        self.outlabeldir = os.path.join(self.out_dir, 'labels')
        if not os.path.exists(self.outlabeldir):
            os.mkdir(self.outlabeldir)

    def collectdata(self,):
        print 'Start Collect Data...'

        train_x_path = os.path.join(self.input_dir, 'train_X.bin')
        train_y_path = os.path.join(self.input_dir, 'train_y.bin')
        test_x_path = os.path.join(self.input_dir, 'test_X.bin')
        test_y_path = os.path.join(self.input_dir, 'test_y.bin')

        train_xf = open(train_x_path, 'rb')
        train_x = np.fromfile(train_xf, dtype=np.uint8)
        train_x = np.reshape(train_x, (-1, 3, 96, 96))
        train_x = np.transpose(train_x, (0, 3, 2, 1))
        train_yf = open(train_y_path, 'rb')
        train_y = np.fromfile(train_yf, dtype=np.uint8)

        test_xf = open(test_x_path, 'rb')
        test_x = np.fromfile(test_xf, dtype=np.uint8)
        test_x = np.reshape(test_x, (-1, 3, 96, 96))
        test_x = np.transpose(test_x, (0, 3, 2, 1))
        test_yf = open(test_y_path, 'rb')
        test_y = np.fromfile(test_yf, dtype=np.uint8)

        idx = np.zeros(10, dtype=np.int)
        for i in xrange(train_x.shape[0]):
            outdir = os.path.join(self.outimgdir, 'train', str(train_y[i]-1))
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            if not self.skipimg:
                transform_and_save(img_arr=train_x[i], output_filename=os.path.join(outdir, str(idx[train_y[i]-1]) + '.jpg'))
            self.trainpairlist[os.path.join('images', 'train', str(train_y[i]-1), str(idx[train_y[i]-1]) + '.jpg')] = \
                os.path.join('labels', str(train_y[i] - 1) + '.txt')
            idx[train_y[i]-1] += 1

        idx = np.zeros(10, dtype=np.int)
        for i in xrange(test_x.shape[0]):
            outdir = os.path.join(self.outimgdir, 'val', str(test_y[i]-1))
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            if not self.skipimg:
                transform_and_save(img_arr=test_x[i],
                                   output_filename=os.path.join(outdir, str(idx[test_y[i]-1]) + '.jpg'))
            self.valpairlist[os.path.join('images', 'val', str(test_y[i]-1), str(idx[test_y[i]-1]) + '.jpg')] = \
                os.path.join('labels', str(test_y[i] - 1) + '.txt')
            idx[test_y[i]-1] += 1

        print 'Finished Collect Data...'

    def write_label(self, ):
        for i, l in enumerate(self.labels):
            sdir = os.path.join(self.outlabeldir, str(i) + '.txt')
            np.savetxt(sdir, [l], '%d')

    def run(self):
        """
        resize images then write manifest files to disk.
        """
        self.write_label()
        self.collectdata()

        records = [(fname, tgt)
                   for fname, tgt in self.trainpairlist.items()]
        np.savetxt(self.manifests['train'], records, fmt='%s,%s')

        records = [(fname, tgt)
                   for fname, tgt in self.valpairlist.items()]
        np.savetxt(self.manifests['val'], records, fmt='%s,%s')


class IngestUnlabeled(object):
    def __init__(self, input_dir, out_dir, target_size=96, skipimg=False):
        np.random.seed(0)
        self.skipimg = skipimg
        self.out_dir = out_dir
        self.input_dir = input_dir

        self.manifests = dict()
        self.manifests = os.path.join(self.out_dir, 'unlabeled-index.csv')

        self.target_size = target_size
        self.trainpairlist = {}

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        self.outimgdir = os.path.join(self.out_dir, 'images')
        if not os.path.exists(self.outimgdir):
            os.mkdir(self.outimgdir)
        self.unlabeldir = os.path.join(self.outimgdir, 'unlabeled')
        if not os.path.exists(self.unlabeldir):
            os.mkdir(self.unlabeldir)

    def collectdata(self,):
        print 'Start Collect Data...'

        train_x_path = os.path.join(self.input_dir, 'unlabeled_X.bin')

        train_xf = open(train_x_path, 'rb')
        train_x = np.fromfile(train_xf, dtype=np.uint8)
        train_x = np.reshape(train_x, (-1, 3, 96, 96))
        train_x = np.transpose(train_x, (0, 3, 2, 1))

        idx = 0
        for i in xrange(train_x.shape[0]):
            if not self.skipimg:
                transform_and_save(img_arr=train_x[i], output_filename=os.path.join(self.unlabeldir, str(idx) + '.jpg'))
            self.trainpairlist[os.path.join('images', 'unlabeled', str(idx) + '.jpg')] = 'labels/11.txt'
            idx += 1

        print 'Finished Collect Data...'

    def write_label(self, ):
        sdir = os.path.join(self.out_dir, 'labels', '11.txt')
        np.savetxt(sdir, [11], '%d')

    def run(self):
        """
        resize images then write manifest files to disk.
        """
        self.write_label()
        self.collectdata()

        records = [(fname, tgt)
                   for fname, tgt in self.trainpairlist.items()]
        np.savetxt(self.manifests, records, fmt='%s,%s')


if __name__ == "__main__":
    parser = ArgParser()
    parser.add_argument('--input_dir', help='Directory to find input',
                        default='/hdd/Dataset/STL10')
    parser.add_argument('--out_dir', help='Directory to write ingested files',
                        default='/home/william/PyProjects/TFcodes/dataset/stl10')
    parser.add_argument('--target_size', type=int, default=96,
                        help='Size in pixels to scale shortest side DOWN to (0 means no scaling)')
    parser.add_argument('--skipImg', type=bool, default=False,
                        help='True to skip processing and copying images')
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    bw = Ingest(input_dir=args.input_dir, out_dir=args.out_dir, target_size=args.target_size, skipimg=args.skipImg)
    # bw = IngestUnlabeled(input_dir=args.input_dir, out_dir=args.out_dir, target_size=args.target_size, skipimg=args.skipImg)
    bw.run()
