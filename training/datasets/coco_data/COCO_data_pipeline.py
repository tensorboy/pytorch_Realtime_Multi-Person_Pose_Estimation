# coding=utf-8
import os

import cv2
import numpy as np

import torch
from training.datasets.coco_data.heatmap import putGaussianMaps
from training.datasets.coco_data.ImageAugmentation import (aug_croppad, aug_flip,
                                                  aug_rotate, aug_scale)
from training.datasets.coco_data.paf import putVecMaps
from training.datasets.coco_data.preprocessing import (inception_preprocess,
                                              rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess)
from torch.utils.data import DataLoader, Dataset

'''
train2014  : 82783 simages
val2014    : 40504 images

first 2644 of val2014 marked by 'isValidation = 1', as our minval dataset.
So all training data have 82783+40504-2644 = 120643 samples
'''

class Cocokeypoints(Dataset):
    def __init__(self, root, mask_dir, index_list, data, inp_size, feat_stride, preprocess='rtpose', transform=None,
                 target_transform=None, params_transform=None):

        self.params_transform = params_transform
        self.params_transform['crop_size_x'] = inp_size
        self.params_transform['crop_size_y'] = inp_size
        self.params_transform['stride'] = feat_stride

        # add preprocessing as a choice, so we don't modify it manually.
        self.preprocess = preprocess
        self.data = data
        self.mask_dir = mask_dir
        self.numSample = len(index_list)
        self.index_list = index_list
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def get_anno(self, meta_data):
        """
        get meta information
        """
        anno = dict()
        anno['dataset'] = meta_data['dataset']
        anno['img_height'] = int(meta_data['img_height'])
        anno['img_width'] = int(meta_data['img_width'])

        anno['isValidation'] = meta_data['isValidation']
        anno['people_index'] = int(meta_data['people_index'])
        anno['annolist_index'] = int(meta_data['annolist_index'])

        # (b) objpos_x (float), objpos_y (float)
        anno['objpos'] = np.array(meta_data['objpos'])
        anno['scale_provided'] = meta_data['scale_provided']
        anno['joint_self'] = np.array(meta_data['joint_self'])

        anno['numOtherPeople'] = int(meta_data['numOtherPeople'])
        anno['num_keypoints_other'] = np.array(
            meta_data['num_keypoints_other'])
        anno['joint_others'] = np.array(meta_data['joint_others'])
        anno['objpos_other'] = np.array(meta_data['objpos_other'])
        anno['scale_provided_other'] = meta_data['scale_provided_other']
        anno['bbox_other'] = meta_data['bbox_other']
        anno['segment_area_other'] = meta_data['segment_area_other']

        if anno['numOtherPeople'] == 1:
            anno['joint_others'] = np.expand_dims(anno['joint_others'], 0)
            anno['objpos_other'] = np.expand_dims(anno['objpos_other'], 0)
        return anno

    def add_neck(self, meta):
        '''
        MS COCO annotation order:
        0: nose	   		1: l eye		2: r eye	3: l ear	4: r ear
        5: l shoulder	6: r shoulder	7: l elbow	8: r elbow
        9: l wrist		10: r wrist		11: l hip	12: r hip	13: l knee
        14: r knee		15: l ankle		16: r ankle

        The order in this work:
        (0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
        5-'left_shoulder' 6-'left_elbow'	    7-'left_wrist'  8-'right_hip'
        9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee'
        13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear'
        17-'left_ear' )
        '''
        our_order = [0, 17, 6, 8, 10, 5, 7, 9,
                     12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        # Index 6 is right shoulder and Index 5 is left shoulder
        right_shoulder = meta['joint_self'][6, :]
        left_shoulder = meta['joint_self'][5, :]
        neck = (right_shoulder + left_shoulder) / 2
        if right_shoulder[2] == 2 or left_shoulder[2] == 2:
            neck[2] = 2
        elif right_shoulder[2] == 1 or left_shoulder[2] == 1:
            neck[2] = 1
        else:
            neck[2] = right_shoulder[2] * left_shoulder[2]

        neck = neck.reshape(1, len(neck))
        neck = np.round(neck)
        meta['joint_self'] = np.vstack((meta['joint_self'], neck))
        meta['joint_self'] = meta['joint_self'][our_order, :]
        temp = []

        for i in range(meta['numOtherPeople']):
            right_shoulder = meta['joint_others'][i, 6, :]
            left_shoulder = meta['joint_others'][i, 5, :]
            neck = (right_shoulder + left_shoulder) / 2
            if (right_shoulder[2] == 2 or left_shoulder[2] == 2):
                neck[2] = 2
            elif (right_shoulder[2] == 1 or left_shoulder[2] == 1):
                neck[2] = 1
            else:
                neck[2] = right_shoulder[2] * left_shoulder[2]
            neck = neck.reshape(1, len(neck))
            neck = np.round(neck)
            single_p = np.vstack((meta['joint_others'][i], neck))
            single_p = single_p[our_order, :]
            temp.append(single_p)
        meta['joint_others'] = np.array(temp)

        return meta

    def remove_illegal_joint(self, meta):
        crop_x = int(self.params_transform['crop_size_x'])
        crop_y = int(self.params_transform['crop_size_y'])
        mask = np.logical_or.reduce((meta['joint_self'][:, 0] >= crop_x,
                                     meta['joint_self'][:, 0] < 0,
                                     meta['joint_self'][:, 1] >= crop_y,
                                     meta['joint_self'][:, 1] < 0))
        # out_bound = np.nonzero(mask)
        # print(mask.shape)
        meta['joint_self'][mask == True, :] = (1, 1, 2)
        if (meta['numOtherPeople'] != 0):
            mask = np.logical_or.reduce((meta['joint_others'][:, :, 0] >= crop_x,
                                         meta['joint_others'][:, :, 0] < 0,
                                         meta['joint_others'][:,
                                                              :, 1] >= crop_y,
                                         meta['joint_others'][:, :, 1] < 0))
            meta['joint_others'][mask == True, :] = (1, 1, 2)

        return meta

    def get_ground_truth(self, meta, mask_miss):

        stride = self.params_transform['stride']
        mode = self.params_transform['mode']
        crop_size_y = self.params_transform['crop_size_y']
        crop_size_x = self.params_transform['crop_size_x']
        num_parts = self.params_transform['np']
        nop = meta['numOtherPeople']
        grid_y = crop_size_y / stride
        grid_x = crop_size_x / stride
        channels = (num_parts + 1) * 2
        heatmaps = np.zeros((int(grid_y), int(grid_x), 19))
        pafs = np.zeros((int(grid_y), int(grid_x), 38))

        mask_miss = cv2.resize(mask_miss, (0, 0), fx=1.0 / stride, fy=1.0 /
                               stride, interpolation=cv2.INTER_CUBIC).astype(
            np.float32)
        mask_miss = mask_miss / 255.
        mask_miss = np.expand_dims(mask_miss, axis=2)

        heat_mask = np.repeat(mask_miss, 19, axis=2)
        paf_mask = np.repeat(mask_miss, 38, axis=2)

        # confidance maps for body parts
        for i in range(18):
            if (meta['joint_self'][i, 2] <= 1):
                center = meta['joint_self'][i, :2]
                gaussian_map = heatmaps[:, :, i]
                heatmaps[:, :, i] = putGaussianMaps(
                    center, gaussian_map, params_transform=self.params_transform)
            for j in range(nop):
                if (meta['joint_others'][j, i, 2] <= 1):
                    center = meta['joint_others'][j, i, :2]
                    gaussian_map = heatmaps[:, :, i]
                    heatmaps[:, :, i] = putGaussianMaps(
                        center, gaussian_map, params_transform=self.params_transform)
        # pafs
        mid_1 = [2, 9, 10, 2, 12, 13, 2, 3, 4,
                 3, 2, 6, 7, 6, 2, 1, 1, 15, 16]

        mid_2 = [9, 10, 11, 12, 13, 14, 3, 4, 5,
                 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]

        thre = 1
        for i in range(19):
            # limb

            count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
            if (meta['joint_self'][mid_1[i] - 1, 2] <= 1 and meta['joint_self'][mid_2[i] - 1, 2] <= 1):
                centerA = meta['joint_self'][mid_1[i] - 1, :2]
                centerB = meta['joint_self'][mid_2[i] - 1, :2]
                vec_map = pafs[:, :, 2 * i:2 * i + 2]
                #                    print vec_map.shape
                pafs[:, :, 2 * i:2 * i + 2], count = putVecMaps(centerA=centerA,
                                                                centerB=centerB,
                                                                accumulate_vec_map=vec_map,
                                                                count=count, params_transform=self.params_transform)
            for j in range(nop):
                if (meta['joint_others'][j, mid_1[i] - 1, 2] <= 1 and meta['joint_others'][j, mid_2[i] - 1, 2] <= 1):
                    centerA = meta['joint_others'][j, mid_1[i] - 1, :2]
                    centerB = meta['joint_others'][j, mid_2[i] - 1, :2]
                    vec_map = pafs[:, :, 2 * i:2 * i + 2]
                    pafs[:, :, 2 * i:2 * i + 2], count = putVecMaps(centerA=centerA,
                                                                    centerB=centerB,
                                                                    accumulate_vec_map=vec_map,
                                                                    count=count, params_transform=self.params_transform)
        # background
        heatmaps[:, :, -
                 1] = np.maximum(1 - np.max(heatmaps[:, :, :18], axis=2), 0.)

        return heat_mask, heatmaps, paf_mask, pafs

    def __getitem__(self, index):
        idx = self.index_list[index]
        img = cv2.imread(os.path.join(self.root, self.data[idx]['img_paths']))
        img_idx = self.data[idx]['img_paths'][-16:-3]
#        print img.shape
        if "COCO_val" in self.data[idx]['dataset']:
            mask_miss = cv2.imread(
                self.mask_dir + 'mask2014/val2014_mask_miss_' + img_idx + 'png', 0)
        elif "COCO" in self.data[idx]['dataset']:
            mask_miss = cv2.imread(
                self.mask_dir + 'mask2014/train2014_mask_miss_' + img_idx + 'png', 0)
#        print self.root + 'mask2014/val2014_mask_miss_' + img_idx + 'png'
        meta_data = self.get_anno(self.data[idx])

        meta_data = self.add_neck(meta_data)

        meta_data, img, mask_miss = aug_scale(
            meta_data, img, mask_miss, self.params_transform)

        meta_data, img, mask_miss = aug_rotate(
            meta_data, img, mask_miss, self.params_transform)

        meta_data, img, mask_miss = aug_croppad(
            meta_data, img, mask_miss, self.params_transform)

        meta_data, img, mask_miss = aug_flip(
            meta_data, img, mask_miss, self.params_transform)

        meta_data = self.remove_illegal_joint(meta_data)

        heat_mask, heatmaps, paf_mask, pafs = self.get_ground_truth(
            meta_data, mask_miss)

        # image preprocessing, which comply the model
        # trianed on Imagenet dataset
        if self.preprocess == 'rtpose':
            img = rtpose_preprocess(img)

        elif self.preprocess == 'vgg':
            img = vgg_preprocess(img)

        elif self.preprocess == 'inception':
            img = inception_preprocess(img)

        elif self.preprocess == 'ssd':
            img = ssd_preprocess(img)

        img = torch.from_numpy(img)
        heatmaps = torch.from_numpy(
            heatmaps.transpose((2, 0, 1)).astype(np.float32))
        heat_mask = torch.from_numpy(
            heat_mask.transpose((2, 0, 1)).astype(np.float32))
        pafs = torch.from_numpy(pafs.transpose((2, 0, 1)).astype(np.float32))
        paf_mask = torch.from_numpy(
            paf_mask.transpose((2, 0, 1)).astype(np.float32))

        return img, heatmaps, heat_mask, pafs, paf_mask

    def __len__(self):
        return self.numSample
