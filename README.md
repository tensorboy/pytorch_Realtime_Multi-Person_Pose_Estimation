# pytorch_Realtime_Multi-Person_Pose_Estimation
This is a pytorch version of Realtime_Multi-Person_Pose_Estimation, origin code is here https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation 

## Introduction
Code repo for reproducing 2017 CVPR Oral paper using pytorch.  

## Contents
1. [Testing](#testing)
2. [Training](#training)

## Testing
- `cd model; sh get_model.sh`
- `cd caffe_to_pytorch; python convert.py`, The converted model have relative error less than 1e-6, and will be located in ./model after converted.
- `pythont picture_demo.py` to run the picture demo.
- `pythont web_demo.py` to run the web demo.

## TODO-Training

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
