#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/testing/CaffeProjects/examples/genre
DATA=/home/testing/CaffeProjects/data/genre
TOOLS=/home/testing/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/genre_train_lmdb \
  $DATA/genre_mean.binaryproto

echo "Done."
