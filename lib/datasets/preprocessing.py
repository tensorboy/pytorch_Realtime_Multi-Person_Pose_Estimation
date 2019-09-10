"""
Provides different utilities to preprocess images.
Args:
image: A np.array representing an image of (h,w,3).

Returns:
A preprocessed image. which dtype is np.float32
and transposed to (3,h,w).

"""

import cv2
import numpy as np


def rtpose_preprocess(image):
    image = image.astype(np.float32)
    image = image / 256. - 0.5
    image = image.transpose((2, 0, 1)).astype(np.float32)

    return image

def inverse_rtpose_preprocess(image):
    image = image.astype(np.float32)
    image = image.transpose((1, 2, 0)).astype(np.float32)    
    image = (image + 0.5) * 256. 
    image = image.astype(np.uint8)


    return image
    
def vgg_preprocess(image):
    image = image.astype(np.float32) / 255.
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = image.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
    return preprocessed_img


def inception_preprocess(image):
    image = image.copy()[:, :, ::-1]
    image = image.astype(np.float32)
    image = image / 128. - 1.
    image = image.transpose((2, 0, 1)).astype(np.float32)

    return image

def inverse_vgg_preprocess(image):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    image = image.transpose((1,2,0))
    
    for i in range(3):
        image[:, :, i] = image[:, :, i] * stds[i]
        image[:, :, i] = image[:, :, i] + means[i]
    image = image.copy()[:,:,::-1]
    image = image*255
    
    return image
    
def inverse_inception_preprocess(image):

    image = image.copy()
    image = image.transpose((1, 2, 0)).astype(np.float32)
    image = image[:, :, ::-1]
    image = (image  + 1.)*128.
    image = image.astype(np.uint8)
    
    return image
    
def ssd_preprocess(image):
    image = image.astype(np.float32)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image -= (104.0, 117.0, 123.0)

    processed_img = rgb_image.astype(np.float32)
    processed_img = processed_img[:, :, ::-1].copy()
    processed_img = processed_img.transpose((2, 0, 1)).astype(np.float32)

    return processed_img


def preprocess(image, mode):
    preprocessors = {
        'rtpose': rtpose_preprocess,
        'vgg': vgg_preprocess,
        'inception': inception_preprocess,
        'ssd': ssd_preprocess
    }
    if mode not in preprocessors:
        return image
    return preprocessors[mode](image)


def put_vec_maps(centerA, centerB, accumulate_vec_map, count, params_transform):
    """Implement Part Affinity Fields
    :param centerA: int with shape (2,), centerA will pointed by centerB.
    :param centerB: int with shape (2,), centerB will point to centerA.
    :param accumulate_vec_map: one channel of paf.
    :param count: store how many pafs overlaped in one coordinate of accumulate_vec_map.
    :param params_transform: store the value of stride and crop_szie_y, crop_size_x
    """
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)

    stride = params_transform['stride']
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    thre = 1  # limb width
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)
    if norm == 0.0:
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
    count[mask] += 1

    mask = count == 0

    count[mask] = 1

    accumulate_vec_map = np.divide(accumulate_vec_map, count[:, :, np.newaxis])
    count[mask] = 0

    return accumulate_vec_map, count


def put_gaussian_maps(center, accumulate_confid_map, params_transform):
    """Implement the generate of every channel of ground truth heatmap.
    :param center: int with shape (2,), every coordinate of person's keypoint.
    :param accumulate_confid_map: one channel of heatmap, which is accumulated,
           np.log(100) is the max value of heatmap.
    :param params_transform: store the value of stride and crop_szie_y, crop_size_x
    """
    LOG_E_100 = 4.6052
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    stride = params_transform['stride']
    sigma = params_transform['sigma']

    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    start = stride / 2.0 - 0.5
    y_range = list(range(int(grid_y)))
    x_range = list(range(int(grid_x)))
    xx, yy = np.meshgrid(x_range, y_range)
    xx = xx * stride + start
    yy = yy * stride + start
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= LOG_E_100
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    accumulate_confid_map += cofid_map
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
    return accumulate_confid_map
