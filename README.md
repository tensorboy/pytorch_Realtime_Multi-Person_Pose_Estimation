## Introduction
Multi Person PoseEstimation By PyTorch

## Results

<p align="left">
<img src="https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/result.gif", width="720">
</p>

[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT) 

## Require
1. [Pytorch](http://pytorch.org/)

## Installation
1. git submodule init && git submodule update

## Demo
- Download [converted pytorch model](https://www.dropbox.com/s/ae071mfm2qoyc8v/pose_model.pth?dl=0).
- Compile the C++ postprocessing: `cd lib/pafprocess; sh make.sh` 
- `python demo/picture_demo.py` to run the picture demo.
- `python demo/web_demo.py` to run the web demo.

## Evalute
- `python evaluate/evaluation.py` to evaluate the model on coco val2017 dataset.
- It should have `mAP 0.653` for the rtpose, previous rtpose have `mAP 0.577` because we do left and right flip for heatmap and PAF for the evaluation. 
c
### Main Results

| model name| mAP |  Inference Time | 
| :---------: | :---------: |:---------: |
|[original rtpose]   | 0.653 |-|

Download link:
[rtpose](https://www.dropbox.com/s/ae071mfm2qoyc8v/pose_model.pth?dl=0)

## Development environment

The code is developed using python 3.6 on Ubuntu 18.04. NVIDIA GPUs are needed. The code is developed and tested using 4 1080ti GPU cards. Other platforms or GPU cards are not fully tested.  

## Quick start

### 1. Preparation

#### 1.1 Prepare the dataset
- `cd training; bash getData.sh` to obtain the COCO 2017 images in `/data/root/coco/images/`, keypoints annotations in `/data/root/coco/annotations/`,
make them look like this:
```
${DATA_ROOT}
|-- coco
    |-- annotations
        |-- person_keypoints_train2017.json
        |-- person_keypoints_val2017.json
    |-- images
        |-- train2017
            |-- 000000000009.jpg
            |-- 000000000025.jpg
            |-- 000000000030.jpg
            |-- ... 
        |-- val2017
            |-- 000000000139.jpg
            |-- 000000000285.jpg
            |-- 000000000632.jpg
            |-- ... 
        

```

### 2. How to train the model
- Modify the data directory in `train/train_VGG19.py` and `python train/train_VGG19.py`

## Related repository
- CVPR'17, [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).

### Network Architecture
- testing architecture
![Teaser?](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/pose.png)

- training architecture
![Teaser?](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/training_structure.png)

## Contributions

All contributions are welcomed. If you encounter any issue (including examples of images where it fails) feel free to open an issue.

## Citation
Please cite the paper in your publications if it helps your research: 

    @InProceedings{cao2017realtime,
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2017}
      }
