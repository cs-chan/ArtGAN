# WikiArt Dataset (Refined)

In order to replicate or to have a fair comparison to our paper, we created a "new" Wikiart dataset that can be downloaded at [this https URL1](https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view?usp=drivesdk) or [this https URL2](http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip) (Size = 25.4Gb). 

<p align="center"> <img src="port.gif" width="30%"> </p>
<p align="center"> Figure 1: Iteration of the portrait generation by ArtGAN++</p>


# Description
Each folder contains information of the dataset used for each tasks (Style, Artist, and Genre classification).

In each folder:

1. {TASK}_class.txt lists the classes and the respective index
2. {TASK}_train.csv lists the images used in the task for training.
3. {TASK}_val.csv lists the images used in the task for validation.

and the csv file can be downloaded at [this https URL](https://drive.google.com/file/d/1uug57zp13wJDwb2nuHOQfR2Odr0hh1a8/view?usp=sharing). In the csv files, each row in the list contains (path/to/image.jpg, class_index)

Note:
1. The WikiArt dataset can be used only for non-commercial research purpose.
2. The images in the WikiArt dataset were obtained from WikiArt.org. The authors are neither responsible for the content nor the meaning of these images.
3. By using the WikiArt dataset, you agree to obey the terms and conditions of [WikiArt.org](https://www.wikiart.org/en/terms-of-use).

# Citation
Shall you use our refined Wikiart dataset, please cite the following paper:

```
@article{artgan2018,
  title={Improved ArtGAN for Conditional Synthesis of Natural Image and Artwork},
  author={Tan, Wei Ren and Chan, Chee Seng and Aguirre, Hernan and Tanaka, Kiyoshi},
  journal={IEEE Transactions on Image Processing},
  volume    = {28},
  number    = {1},
  pages     = {394--409},
  year      = {2019},
  url       = {https://doi.org/10.1109/TIP.2018.2866698},
  doi       = {10.1109/TIP.2018.2866698}
}
```
