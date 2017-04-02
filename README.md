# pytorch_Realtime_Multi-Person_Pose_Estimation
This is a pytorch version of Realtime_Multi-Person_Pose_Estimation, origin code is here https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation 

## Introduction
Code repo for reproducing 2017 CVPR Oral paper using pytorch.  

## Results

<p align="left">
<img src="https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/result1.gif", width="720">
</p>

<p align="left">
<img src="https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/result2.gif", width="720">
</p>

## Contents
1. [Testing](#testing)
2. [Training](#training)

## Require
1. [Pytorch](http://pytorch.org/)
2. [Caffe](http://caffe.berkeleyvision.org/) is required if you want convert caffe model to a pytorch model.

## Testing
- `cd model; sh get_model.sh` to download caffe model or download converted pytorch model(https://www.dropbox.com/s/ae071mfm2qoyc8v/pose_model.pth?dl=0).
- `cd caffe_to_pytorch; python convert.py` to convert a trained caffe model to pytorch model. The converted model have relative error less than 1e-6, and will be located in `./model` after converted.
- `pythont picture_demo.py` to run the picture demo.
- `pythont web_demo.py` to run the web demo.

## Training
- Run `cd training; bash getData.sh` to obtain the COCO images in `dataset/COCO/images/`, keypoints annotations in `dataset/COCO/annotations/` and [COCO official toolbox](https://github.com/pdollar/coco) in `dataset/COCO/coco/`. 

### Network Architecture
![Teaser?](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/pose.png)

## Citation
Please cite the paper in your publications if it helps your research:    
	  
    @inproceedings{wei2016cpm,
      author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Convolutional pose machines},
      year = {2016}
      }
