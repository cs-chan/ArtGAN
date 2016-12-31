#!/usr/bin/env sh

/home/testing/caffe/build/tools/caffe test \
    --model=models/alexnet_finetune_style/deploy.prototxt \
    --weights=models/alexnet_finetune_style/caffe_alexnet_style_train.caffemodel \
    --iterations=215 --gpu=0
