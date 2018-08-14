# coding=utf-8
"""Implement Part Affinity Fields
:param centerA: int with shape (2,), centerA will pointed by centerB.
:param centerB: int with shape (2,), centerB will point to centerA.
:param accumulate_vec_map: one channel of paf.
:param count: store how many pafs overlaped in one coordinate of accumulate_vec_map.
:param params_transform: store the value of stride and crop_szie_y, crop_size_x                 
"""
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage


def putVecMaps(centerA, centerB, accumulate_vec_map, count, params_transform):
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)

    stride = params_transform['stride']
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    thre = params_transform['limb_width']   # limb width
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)
    if (norm == 0.0):
        # print 'limb is too short, ignore it...'
        return accumulate_vec_map, count
    limb_vec_unit = limb_vec / norm
    # print 'limb unit vector: {}'.format(limb_vec_unit)

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    vec_map = np.copy(accumulate_vec_map) * 0.0
    vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]

    mask = np.logical_or.reduce(
        (np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))

    accumulate_vec_map = np.multiply(
        accumulate_vec_map, count[:, :, np.newaxis])
    accumulate_vec_map += vec_map
    count[mask == True] += 1

    mask = count == 0

    count[mask == True] = 1

    accumulate_vec_map = np.divide(accumulate_vec_map, count[:, :, np.newaxis])
    count[mask == True] = 0

    return accumulate_vec_map, count
