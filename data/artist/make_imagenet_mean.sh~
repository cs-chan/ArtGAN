#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/testing/CaffeProjects/examples/artist
DATA=/home/testing/CaffeProjects/data/artist
TOOLS=/home/testing/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/artist_train_lmdb \
  $DATA/artist_mean.binaryproto

echo "Done."
