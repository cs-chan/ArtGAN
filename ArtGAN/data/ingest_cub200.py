from configargparse import ArgParser
from PIL import Image
import logging
import numpy as np
import os
import shutil


def transform_and_save(img_path, target_size, output_filename, skip=False):
    """
    Takes an image and
    optionally transforms it and then writes it out to output_filename
    """
    if skip and os.path.exists(output_filename):
        return
    img = Image.open(img_path)
    width, height = img.size

    # Take the smaller image dimension down to target_size
    # while retaining aspect_ration. Otherwise leave it alone
    if width < height:
        if width > target_size:
            scale_factor = float(target_size) / width
            width = target_size
            height = int(height*scale_factor)
    else:
        if height > target_size:
            scale_factor = float(target_size) / height
            height = target_size
            width = int(width*scale_factor)
    if img.size[0] != width or img.size[1] != height:
        img = img.resize((width, height), resample=Image.LANCZOS)
        img.save(output_filename, quality=100)
    else:
        # Avoid recompression by saving file out directly without transformation
        shutil.copy(img_path, output_filename)
    assert (os.stat(output_filename).st_size > 0), "{} has size 0".format(output_filename)


class Ingest(object):

    def __init__(self, input_dir, out_dir, target_size=256, skipimg=False):
        np.random.seed(0)
        self.skipimg = skipimg
        self.out_dir = out_dir
        self.input_dir = input_dir
        self.input_img_dir = os.path.join(input_dir, 'images')

        self.manifests = dict()
        for setn in ('train', 'val'):
            self.manifests[setn] = os.path.join(self.out_dir, '{}-index.csv'.format(setn))

        self.target_size = target_size
        self.ntrain = []

        self.trainpairlist = {}
        self.valpairlist = {}
        self.labels = range(200)

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        self.outimgdir = os.path.join(self.out_dir, 'images')
        if not os.path.exists(self.outimgdir):
            os.mkdir(self.outimgdir)

        self.outlabeldir = os.path.join(self.out_dir, 'labels')
        if not os.path.exists(self.outlabeldir):
            os.mkdir(self.outlabeldir)

    def collectdata(self,):
        print 'Start Collect Data...'
        with open(self.input_dir + '/images.txt', 'rb') as f:
            img_list = f.readlines()
        with open(self.input_dir + '/image_class_labels.txt', 'rb') as f:
            img_labels = f.readlines()
        with open(self.input_dir + '/train_test_split.txt', 'rb') as f:
            img_split = f.readlines()

        for img, label, trainval in zip(img_list, img_labels, img_split):
            img_pathsemi = img.split()[1]
            lab = int(label.split()[1]) - 1
            tv = int(trainval.split()[1])

            sdir = os.path.join(self.outimgdir, img_pathsemi.split('/')[0])
            if not os.path.exists(sdir):
                os.mkdir(sdir)

            imgpath = os.path.join(self.input_img_dir, img_pathsemi)
            outpath = os.path.join(self.outimgdir, img_pathsemi)
            transform_and_save(img_path=imgpath, output_filename=outpath, target_size=self.target_size, skip=self.skipimg)

            if tv:
                self.trainpairlist[os.path.join('images', img_pathsemi)] = os.path.join('labels', str(lab) + '.txt')
            else:
                self.valpairlist[os.path.join('images', img_pathsemi)] = os.path.join('labels', str(lab) + '.txt')

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


if __name__ == "__main__":
    parser = ArgParser()
    parser.add_argument('--input_dir', help='Directory to find input',
                        default='/hdd/Dataset/CUB200/CUB_200_2011')
    parser.add_argument('--out_dir', help='Directory to write ingested files',
                        default='/home/william/PyProjects/TFcodes/dataset/cub200')
    parser.add_argument('--target_size', type=int, default=256,
                        help='Size in pixels to scale shortest side DOWN to (0 means no scaling)')
    parser.add_argument('--ratio', type=float, default=0.3,
                        help='Percentage of dataset to be used for validation')
    parser.add_argument('--skipImg', type=bool, default=True,
                        help='True to skip processing and copying images')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    bw = Ingest(input_dir=args.input_dir, out_dir=args.out_dir, target_size=args.target_size,
                skipimg=args.skipImg)
    bw.run()
