# ArtGAN

This is implementation of our ICIP2017 paper with titled [ArtGAN: Artwork Synthesis with Conditional Categorial GANs](https://arxiv.org/abs/1702.03410).

# Citation
This repository contains codes for the following paper (under review):

```
@article{tan2017learning,
  title={Learning a Generative Adversarial Network for High Resolution Artwork Synthesis},
  author={Tan, Wei Ren and Chan, Chee Seng and Aguirre, Hernan and Tanaka, Kiyoshi},
  journal={arXiv preprint arXiv:1708.09533},
  year={2017}
}
```
which is an extension to the following paper (ICIP 2017): 
```
@article{tan2017artgan,
  title={ArtGAN: Artwork Synthesis with Conditional Categorial GANs},
  author={Tan, Wei Ren and Chan, Chee Seng and Aguirre, Hernan and Tanaka, Kiyoshi},
  journal={arXiv preprint arXiv:1702.03410},
  year={2017}
}
```

# Prerequisites
- Python 2.7
- [Tensorflow](https://github.com/tensorflow/tensorflow.git)
- (Optional) [Nervana's Systems neon](https://github.com/NervanaSystems/neon.git)
- (Optional) [Nervana's Systems aeon](https://github.com/NervanaSystems/aeon.git)

\* Neon and aeon are required to load data. If other data loader is used, neon and aeon are not required. But, make sure that data format is 'NCHW'.

# Trained models

Each link below is the best trained model for the corresponding dataset:

- CIFAR-10 - http://www.cs-chan.com/source/ArtGAN/CIFAR64GANAE.zip

- STL-10 - http://www.cs-chan.com/source/ArtGAN/STL128GANAE.zip

- Flowers - http://www.cs-chan.com/source/ArtGAN/Flower128GANAE.zip

- CUB-200 - http://www.cs-chan.com/source/ArtGAN/CUB128GANAE.zip

- Wikiart Artist - http://www.cs-chan.com/source/ArtGAN/Artist128GANAE.zip

- Wikiart Genre - http://www.cs-chan.com/source/ArtGAN/Genre128GANAE.zip

- Wikiart Style - http://www.cs-chan.com/source/ArtGAN/Style128GANAE.zip

# Feedback
Suggestions and opinions of this work (both positive and negative) are greatly welcome. Please contact the authors by sending email to Wei Ren Tan at `wrtan.edu at gmail.com` or Chee Seng Chan at `cs.chan at um.edu.my`

# License
BSD-3, see LICENSE file for details.
