# Ceci n'est pas une pipe: A deep convolutional network for fine-art paintings classification

Released on December 20, 2016.

## Description

This is the implementation of our ICIP-16 work with titled -- [Ceci n'est pas une pipe: A Deep Convolutional Network for Fine-art Paintings Classification](http://web.fsktm.um.edu.my/~cschan/doc/ICIP2016.pdf).

<img src="ICIP2016.gif" width="25%">

## Citation
If you find this code useful for your research, please cite
```
@inproceedings{TanICIP16,
  author    = {Wei Ren Tan and
               Chee Seng Chan and
               Hern{\'{a}}n E. Aguirre and
               Kiyoshi Tanaka},
  title     = {Ceci n'est pas une pipe: {A} deep convolutional network for fine-art
               paintings classification},
  booktitle = {IEEE International Conference on Image Processing {ICIP}},
  pages     = {3703--3707},
  year      = {2016},
  doi       = {10.1109/ICIP.2016.7533051},
}
```

## Dependency
The codes are based on [caffe](https://github.com/BVLC/caffe).
<!---
This repository does not include the Wikiart dataset used. 
--->
For the list of the styles, artists, and genres used, please refer to our [ICIP-16 supplementary material](http://web.fsktm.um.edu.my/~cschan/doc/ICIP2016_supp.pdf).

For the details of the WikiArt dataset used in our ICIP-16 paper, please refer to the [WikiArt Dataset](https://github.com/cs-chan/ICIP2016-PC/tree/master/WikiArt%20Dataset) folder.

<!---
For the full list of paintings used, please refer to [Saleh & Elgammal (2015). Large-scale Classification of Fine-Art Paintings: Learning The Right Metric on The Right Feature](https://arxiv.org/pdf/1505.00855v1.pdf).
--->

## Installation and Running

1. Users are required to install Caffe Library from https://github.com/BVLC/caffe. 
2. Users can required to download the Wikiart Dataset provided at [this https URL](https://github.com/cs-chan/ICIP2016-PC/tree/master/WikiArt%20Dataset) for a fair comparison to our ICIP-16 paper.
3. Please choose one of the following options: 

Option 1: 
- These are the Alexnet models pre-trained on ImageNet classification task and fine-tuned on 3 different classes in the Wikiart dataset.
```
Artist
```
- [Alexnet_Artist_train.caffemodel](http://web.fsktm.um.edu.my/~cschan/source/ICIP2016/Alexnet_genre_finetune.caffemodel.zip)

```
Genre
```
- [Alexnet_Genre_train.caffemodel](http://web.fsktm.um.edu.my/~cschan/source/ICIP2016/Alexnet_genre_finetune.caffemodel.zip)

```
Style
```
- [Alexnet_Style_train.caffemodel](http://web.fsktm.um.edu.my/~cschan/source/ICIP2016/Alexnet_style_finetune.caffemodel.zip)


Option 2: 
- New model can be trained via the following command (using artist as example):
```
>> ./path/to/folder/alexnet_finetune_artist/train_caffenet.sh
```

- To compute the accuracy of a given testing set:
```
>> ./path/to/folder/alexnet_finetune_artist/test_caffenet.sh
```

*Note that users are expected to modify the corresponding files to the correct path to work properly. 

Enjoy! :P

## Feedback
Suggestions and opinions of this work (both positive and negative) are greatly welcome. Please contact the authors by sending email to
`wrtan.edu at gmail.com`or `cs.chan at um.edu.my`.

## License
The project is open source under BSD-3 license (see the `LICENSE` file). Codes can be used freely only for academic purpose.
