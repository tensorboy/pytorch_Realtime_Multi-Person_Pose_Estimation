# pytorch_Realtime_Multi-Person_Pose_Estimation
This is a pytorch version of Realtime_Multi-Person_Pose_Estimation, origin code is here https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation 

## Introduction
Code repo for reproducing 2017 CVPR Oral paper using pytorch.  

## Results

<p align="left">
<img src="https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/result1.gif", width="720">
</p>

[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT) 

## Require
1. [Pytorch](http://pytorch.org/)
2. [Caffe](http://caffe.berkeleyvision.org/) is required if you want convert caffe model to a pytorch model.
3. pip install pycocotools
4. pip install tensorboardX
5. pip install torch-encoding


## Demo
- Download [converted pytorch model](https://www.dropbox.com/s/ae071mfm2qoyc8v/pose_model.pth?dl=0).
- `cd network/caffe_to_pytorch; python convert.py` to convert a trained caffe model to pytorch model. The converted model have relative error less than 1e-6, and will be located in `./network/weight` after convert.
- Or use the model trained from scratch in [this repo](https://www.dropbox.com/s/5v654d2u65fuvyr/pose_model_scratch.pth?dl=0), which has better accuracy on the validataion set.
- `python demo/picture_demo.py` to run the picture demo.
- `python demo/web_demo.py` to run the web demo.

## Evalute
- `python evaluate/evaluation.py` to evaluate the model on [images seperated by the original author](https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose/blob/master/image_info_val2014_1k.txt)
- It should have `mAP 0.598` for the original rtpose, original repo have `mAP 0.577` because we do left and right flip for heatmap and PAF for the evaluation. 
c
### Pretrained Models & Performance on the dataset split by the original rtpose.
[rtpose original](https://www.dropbox.com/s/ae071mfm2qoyc8v/pose_model.pth?dl=0), [trained from scratch](https://www.dropbox.com/s/5v654d2u65fuvyr/pose_model_scratch.pth?dl=0) (Notice the preprocessing is different for different models)

|   Reported on paper (VGG19)| mAP in this repo (VGG19)| Trained from scratch in this repo| 
|  :------:     | :---------: | :---------: |
|   0.577      | 0.598     |  **0.614** |


## Training
- `cd training; bash getData.sh` to obtain the COCO images in `dataset/COCO/images/`, keypoints annotations in `dataset/COCO/annotations/`
- Download the mask of the unlabeled person at [Dropbox](https://www.dropbox.com/s/bd9ty7b4fqd5ebf/mask.tar.gz?dl=0)
- Download the official training format at [Dropbox](https://www.dropbox.com/s/0sj2q24hipiiq5t/COCO.json?dl=0)
- `python train_VGG19.py --batch_size 100 --logdir {where to store tensorboardX logs}`
- `python train_ShuffleNetV2.py --batch_size 160 --logdir {where to store tensorboardX logs}`
- `python train_SH.py --batch_size 64 --lr 0.1 --logdir {where to store tensorboardX logs}`
## Related repository
- CVPR'16, [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release).
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
	  
