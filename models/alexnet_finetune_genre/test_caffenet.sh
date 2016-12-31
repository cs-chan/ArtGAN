#!/usr/bin/env sh

/home/testing/caffe/build/tools/caffe test \
    --model=models/alexnet_finetune_genre/deploy.prototxt \
    --weights=models/alexnet_finetune_genre/caffe_alexnet_genre_train.caffemodel \
    --iterations=172 --gpu=0
