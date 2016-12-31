#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/artist
DATA=data/artist
TOOLS=caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/artist_train_lmdb \
  $DATA/artist_mean.binaryproto

echo "Done."
