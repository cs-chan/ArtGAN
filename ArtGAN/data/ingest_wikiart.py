from configargparse import ArgParser
from PIL import Image
import logging
import numpy as np
import os
import shutil
import csv


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

    def __init__(self, input_dir, out_dir, csv_file, target_size=256, skipimg=False):
        """
        :param input_dir: input directory
        :param out_dir: output directory
        :param csv_file: style or artist or genre
        :param target_size: resized image size
        :param skipimg: skip saving image if true and the image exists
        """
        np.random.seed(0)
        self.skipimg = skipimg
        self.out_dir = out_dir
        self.input_dir = input_dir
        self.input_img_dir = os.path.join(input_dir, 'style')
        self.input_csv_dir = os.path.join(input_dir, 'wikiart_csv')

        self.csv_train = self.input_csv_dir + '/' + csv_file + '_train.csv'
        self.csv_val = self.input_csv_dir + '/' + csv_file + '_val.csv'

        self.manifests = dict()
        for setn in ('train', 'val'):
            self.manifests[setn] = os.path.join(self.out_dir, csv_file + '-{}-index.csv'.format(setn))

        self.target_size = target_size
        self.ntrain = []

        self.trainpairlist = {}
        self.valpairlist = {}
        if csv_file is 'style':
            self.labels = range(27)
        elif csv_file is 'genre':
            self.labels = range(10)
        elif csv_file is 'artist':
            self.labels = range(23)
        else:
            raise ValueError('csv_file must be either [style, genre, artist]')

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
        csvtrainfile = open(self.csv_train, 'rb')
        trainlist = csv.reader(csvtrainfile, delimiter=',')
        csvvalfile = open(self.csv_val, 'rb')
        vallist = csv.reader(csvvalfile, delimiter=',')

        for row in trainlist:
            img_name = row[0]
            label = row[1]

            imgpath = os.path.join(self.input_img_dir, img_name)
            outpath = os.path.join(self.outimgdir, img_name)
            sdir = os.path.join(self.outimgdir, imgpath.split('/')[-2])
            if not os.path.exists(sdir):
                os.mkdir(sdir)
            transform_and_save(img_path=imgpath, output_filename=outpath, target_size=self.target_size, skip=self.skipimg)

            self.trainpairlist[os.path.join('images', img_name)] = os.path.join('labels', str(label) + '.txt')

        for row in vallist:
            img_name = row[0]
            label = row[1]

            imgpath = os.path.join(self.input_img_dir, img_name)
            outpath = os.path.join(self.outimgdir, img_name)
            transform_and_save(img_path=imgpath, output_filename=outpath, target_size=self.target_size, skip=self.skipimg)

            self.valpairlist[os.path.join('images', img_name)] = os.path.join('labels', str(label) + '.txt')

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
                        default='/hdd/Dataset/wikiart')
    parser.add_argument('--out_dir', help='Directory to write ingested files',
                        default='/home/william/PyProjects/TFcodes/dataset/wikiart')
    parser.add_argument('--target_size', type=int, default=256,
                        help='Size in pixels to scale shortest side DOWN to (0 means no scaling)')
    parser.add_argument('--ratio', type=float, default=0.3,
                        help='Percentage of dataset to be used for validation')
    parser.add_argument('--skipImg', type=bool, default=True,
                        help='True to skip processing and copying images')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    bw = Ingest(input_dir=args.input_dir, out_dir=args.out_dir, csv_file='artist', target_size=args.target_size,
                skipimg=args.skipImg)
    bw.run()
