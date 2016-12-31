# ICIP2016-Fine Art Painting Classification

Released on December 20, 2016

## Description

This is the implementation of our ICIP-16 work with titled [Ceci n'est pas une pipe: A Deep Convolutional Network for Fine-art Paintings Classification](http://www.cs-chan.com/doc/ICIP2016.pdf).

The project is open source under BSD-3 license (see the `LICENSE` file). Codes can be used freely only for academic purpose.

## Citation
If you use the codes as part of your research project, please cite our work as follows:
```
@inproceedings{TanCAT16,
  author    = {Wei Ren Tan and
               Chee Seng Chan and
               Hern{\'{a}}n E. Aguirre and
               Kiyoshi Tanaka},
  title     = {Ceci n'est pas une pipe: {A} deep convolutional network for fine-art
               paintings classification},
  booktitle = {2016 {IEEE} International Conference on Image Processing, {ICIP} 2016,
               Phoenix, AZ, USA, September 25-28, 2016},
  pages     = {3703--3707},
  year      = {2016},
  url       = {http://dx.doi.org/10.1109/ICIP.2016.7533051},
  doi       = {10.1109/ICIP.2016.7533051},
}
```

## Dependency
The codes are based on [caffe](https://github.com/BVLC/caffe).

This repository does not include the Wikiart dataset used. For the list of the styles, artists, and genres used, please refer to the [ICIP supplementary material](http://www.cs-chan.com/doc/ICIP2016_supp.pdf)

For the full list of paintings used, please refer to the [Saleh, B., & Elgammal, A. (2015). Large-scale Classification of Fine-Art Paintings:Learning The Right Metric on The Right Feature](https://arxiv.org/pdf/1505.00855v1.pdf)

## Installation and Running

Users are required to install Caffe Library from https://github.com/BVLC/caffe. 

This is the Alexnet model pre-trained on ImageNet classification task and fine-tuned on 431 car models in Wikiart dataset.
[alexnet_artist_train.caffemodel](https://arxiv.org/pdf/1505.00855v1.pdf)

1. New model can be trained via the following command (using artist as example):
```matlab
>> ./path/to/folder/alexnet_finetune_artist/train_caffenet.sh
```

2. To compute the accuracy of a given testing set:
```matlab
>> ./path/to/folder/alexnet_finetune_artist/test_caffenet.sh
```

*Note that users are expected to modify the corresponding files to the correct path to work properly. 

Enjoy! :P

## Feedback
Suggestions and opinions of this work (both positive and negative) are greatly welcome. Please contact the authors by sending email to
`wrtan.edu at gmail.com`or `cs.chan at um.edu.my`.

## License
BSD-3, see `LICENSE` file for details.


