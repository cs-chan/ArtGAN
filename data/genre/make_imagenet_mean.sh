#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/genre
DATA=data/genre
TOOLS=caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/genre_train_lmdb \
  $DATA/genre_mean.binaryproto

echo "Done."
