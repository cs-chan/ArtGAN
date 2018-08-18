# ArtGAN/Improved ArtGAN

Updated on August 18, 2018 (Improved ArtGan models are included)

Release on May 20, 2018.

# Description

This is implementation of our TIP-18 (Accepted) work with titled [Improved ArtGAN for Conditional Synthesis of Natural Image and Artwork](coming soon). The preliminary version of this work was presented earlier in ICIP-17 with titled [ArtGAN: Artwork Synthesis with Conditional Categorial GANs](https://arxiv.org/abs/1702.03410). 

The details of the model settings and training scheme can be found at [this PDF](coming soon).

<img src="artgan.gif" width="10%">

# Citation
This repository contains codes for the following paper:

```
@article{tan2017learning,
  title={Improved ArtGAN for Conditional Synthesis of Natural Image and Artwork},
  author={Tan, Wei Ren and Chan, Chee Seng and Aguirre, Hernan and Tanaka, Kiyoshi},
  journal={IEEE Transactions on Image Processing},
  year={Accepted}
}
```
which is an extension to the following paper (ICIP 2017): 
```
@inproceedings{TanCAT17,
  author    = {Tan, Wei Ren and Chan, Chee Seng and Aguirre, Hernan and Tanaka, Kiyoshi},
  title     = {ArtGAN: Artwork synthesis with conditional categorical GANs},
  booktitle = {{IEEE} International Conference on Image Processing {ICIP}},
  pages     = {3760--3764},
  year      = {2017},
  doi       = {10.1109/ICIP.2017.8296985},
}
```

# Prerequisites
- Python 2.7
- [Tensorflow](https://github.com/tensorflow/tensorflow.git)
- (Optional) [Nervana's Systems neon](https://github.com/NervanaSystems/neon.git)
- (Optional) [Nervana's Systems aeon](https://github.com/NervanaSystems/aeon.git)

\* Neon and aeon are required to load data. If other data loader is used, neon and aeon are not required. But, make sure that data format is 'NCHW'.

# Trained models

Each link below is the best trained model used in the T-IP paper for the corresponding dataset:

- CIFAR-10 - available at [this https URL](http://www.cs-chan.com/source/ArtGAN/CIFAR64GANAE.zip)

- STL-10 - available at [this https URL](http://www.cs-chan.com/source/ArtGAN/STL128GANAE.zip)

- Flowers - available at [this https URL](http://www.cs-chan.com/source/ArtGAN/Flower128GANAE.zip)

- CUB-200 - available at [this https URL](http://www.cs-chan.com/source/ArtGAN/CUB128GANAE.zip)

- Wikiart Artist - available at [this https URL](http://www.cs-chan.com/source/ArtGAN/Artist128GANAE.zip)

- Wikiart Genre - available at [this https URL](http://www.cs-chan.com/source/ArtGAN/Genre128GANAE.zip)

- Wikiart Style - available at [this https URL](http://www.cs-chan.com/source/ArtGAN/Style128GANAE.zip)

# Feedback
Suggestions and opinions of this work (both positive and negative) are greatly welcome. Please contact the authors by sending email to Wei Ren Tan at `wrtan.edu at gmail.com` or Chee Seng Chan at `cs.chan at um.edu.my`

# License
BSD-3, see LICENSE file for details.
