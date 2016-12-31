This repository contains code and trained models for the paper: 

Tan, W. R., Chan, C. S., Aguirre, H. E., & Tanaka, K. (2016, September). 
Ceci n'est pas une pipe: A deep convolutional network for fine-art paintings classification. 
In Image Processing (ICIP), 2016 IEEE International Conference on (pp. 3703-3707). IEEE.

Please cite this paper if you use this code as part of your published work. 

This code requires CAFFE to be installed. For more information about CAFFE installation, 
please visit their website: http://caffe.berkeleyvision.org/

This repository does not include the Wikiart dataset used. For the list of the styles, artists,
and genres used, please refer to the supplementary material available in the following website: http://www.cs-chan.com/publication.html
For the full list of paintings used, please contact the authors of the following paper:

Saleh, B., & Elgammal, A. (2015). Large-scale Classification of Fine-Art Paintings: 
Learning The Right Metric on The Right Feature. arXiv preprint arXiv:1505.00855.

New model can be trained via the following command (using artist as example):

./path/to/folder/alexnet_finetune_artist/train_caffenet.sh

To compute the accuracy of a given testing set:

./path/to/folder/alexnet_finetune_artist/test_caffenet.sh

*Note that users are expected to modify the corresponding files to the correct path to work properly. 

Users may contact us at either of the following email for any question regarding our paper. 

Wei Ren, Tan: wrtan.edu at gmail.com
Chee Seng, Chan: cs.chan at um.edu.my 
