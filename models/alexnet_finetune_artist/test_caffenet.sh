#!/usr/bin/env sh

/home/testing/caffe/build/tools/caffe test \
    --model=models/alexnet_finetune_artist/deploy.prototxt \
    --weights=models/alexnet_finetune_artist/caffe_alexnet_artist_train.caffemodel \
    --iterations=50 --gpu=0
