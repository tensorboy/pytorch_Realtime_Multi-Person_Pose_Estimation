import copy
import os
import sys
import time

import cv2
import numpy as np
import json
import torch
from evaluate.beaneval import BEANeval
from lib.datasets import transforms
from lib.datasets.datasets import get_soybean_dataset
from lib.datasets.preprocessing import (inception_preprocess,
                                        rtpose_preprocess,
                                        ssd_preprocess, vgg_preprocess)
from lib.network import im_transform
from lib.config import cfg, update_config
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans, draw_pods
from lib.utils.paf_to_pods import paf_to_pods_cpp

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve

def eval_bean(outputs, ann_paths):
    """Evaluate images on Soybean test set
    :param outputs: list of dictionaries, the models' processed outputs
    :param ann_paths: list, all the annotation paths in the validation set
    :returns : float, the mAP score
    """
    with open('results.json', 'w') as f:
        json.dump(outputs, f)
    beanGt = ann_paths  # load annotations
    beanDt = loadRes('results.json')  # load model outputs

    # running evaluation
    beanEval = BEANeval(beanGt, beanDt)
    beanEval.evaluate()
    beanEval.accumulate()
    beanEval.summarize()
    # os.remove('results.json')
    # return Average Precision
    return beanEval.stats[0]


def add_bbox_info(anns):
    print('Loading and preparing results...')
    tic = time.time()
    anns = copy.deepcopy(anns)
    for id, ann in enumerate(anns):
        s = ann['keypoints']
        x = s[0::2]
        y = s[1::2]
        x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
        ann['area'] = (x1-x0)*(y1-y0)
        ann['bbox'] = [x0, y0, x1-x0, y1-y0]
    print('DONE (t={:0.2f}s)'.format(time.time() - tic))
    return anns


def loadRes(resFile):
    with open(resFile) as f:
        anns = json.load(f)
    assert type(anns) == list, 'results in not an array of objects'
    anns = add_bbox_info(anns)
    return anns

def get_outputs(img, model, preprocess):
    """Computes the averaged heatmap and paf for the given image
    :param multiplier:
    :param origImg: numpy array, the image being processed
    :param model: pytorch model
    :returns: numpy arrays, the averaged paf and heatmap
    """
    inp_size = cfg.DATASET.IMAGE_SIZE

    # padding
    im_croped, im_scale, real_shape = im_transform.crop_with_factor(
        img, inp_size, factor=cfg.MODEL.DOWNSAMPLE, is_ceil=True)

    if preprocess == 'rtpose':
        im_data = rtpose_preprocess(im_croped)

    elif preprocess == 'vgg19':
        im_data = vgg_preprocess(im_croped)

    elif preprocess == 'inception':
        im_data = inception_preprocess(im_croped)

    elif preprocess == 'ssd':
        im_data = ssd_preprocess(im_croped)

    print('im_data.shape:', im_data.shape)
    batch_images = np.expand_dims(im_data, 0)

    # several scales as a batch
    batch_var = torch.from_numpy(batch_images).cuda().float()
    predicted_outputs, _ = model(batch_var)
    output1, output2 = predicted_outputs[-2], predicted_outputs[-1]

    paf = output1.cpu().data.numpy().transpose(0, 2, 3, 1)[0]
    heatmap = output2.cpu().data.numpy().transpose(0, 2, 3, 1)[0]

    # for new openpose model
    # paf = output2[0].cpu().data.numpy().transpose(0, 2, 3, 1)[0]
    # heatmap = output2[1].cpu().data.numpy().transpose(0, 2, 3, 1)[0]

    return paf, heatmap, im_scale


def append_result(image_name, pods, upsample_keypoints, outputs):
    """Build the outputs to be evaluated
    :param image_id: int, the id of the current image
    :param person_to_joint_assoc: numpy array of joints associations
    :param joint_list: list, list of joints
    :param outputs: list of dictionaries with the following keys: image_id,
                    category_id, keypoints, score
    """
    for pod in pods:
        one_result = {
            "image_name": '',
            "category_id": 1,
            "keypoints": [],
            "score": 0
        }
        one_result["image_name"] = image_name
        keypoints = np.zeros((5, 3))

        all_scores = []
        for i in range(cfg.MODEL.NUM_KEYPOINTS):
            if i not in pod.body_parts.keys():
                keypoints[i, 0] = 0
                keypoints[i, 1] = 0
                keypoints[i, 2] = 0
            else:
                body_part = pod.body_parts[i]
                center = (body_part.x * upsample_keypoints[1] + 0.5, body_part.y * upsample_keypoints[0] + 0.5)
                keypoints[i, 0] = center[0]
                keypoints[i, 1] = center[1]
                keypoints[i, 2] = 1

                score = pod.body_parts[i].score
                all_scores.append(score)

        one_result["score"] = 1.
        one_result["keypoints"] = list(keypoints.reshape(15))
        outputs.append(one_result)


def run_eval(image_dir, vis_dir, model, preprocess):
    """Run the evaluation on the test set and report mAP score
    :param model: the model to test
    :returns: float, the reported mAP score
    """
    img_paths, ann_paths = get_soybean_dataset(image_dir)
    print("Total number of validation images {}".format(len(img_paths)))

    # iterate all val images
    outputs = []
    print("Processing Images in validation set")
    for i in range(len(img_paths)):
        if i % 10 == 0 and i != 0:
            print("Processed {} images".format(i))
        # load image and annotations
        img_path = img_paths[i]
        ann_path = ann_paths[i]
        oriImg = cv2.imread(img_path)
        f = open(ann_path)
        anno_file = json.load(f)
        anns = transforms.NormalizeBean().normalize_annotations(anno_file)
        file_name = anns['image_name']
        print('file_name:', file_name)

        # Get the shortest side of the image (either height or width)
        shape_dst = np.min(oriImg.shape[0:2])

        # Get results of original image
        paf, heatmap, scale_img = get_outputs(oriImg, model, preprocess)
        print('paf:', paf.shape)
        print('heatmap:', heatmap.shape)
        pods = paf_to_pods_cpp(heatmap, paf, cfg)
        print('pods:', pods)
        out = draw_pods(oriImg, pods)

        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        vis_path = os.path.join(vis_dir, file_name)
        cv2.imwrite(vis_path, out)
        # subset indicated how many peoples found in this image.
        upsample_keypoints = (
        heatmap.shape[0] * cfg.MODEL.DOWNSAMPLE / scale_img, heatmap.shape[1] * cfg.MODEL.DOWNSAMPLE / scale_img)
        append_result(os.path.basename(img_paths[i]), pods, upsample_keypoints, outputs)

    # Eval and show the final result!
    return eval_bean(outputs=outputs, ann_paths=ann_paths)




