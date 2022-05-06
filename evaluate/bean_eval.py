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
from lib.utils.common import draw_pods, draw_bbox
from lib.utils.paf_to_pods import paf_to_pods_cpp


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


def add_bbox_info(anns, margin=15, shape=None):
    """
    compute bbox information and add it to the annotation
    :param anns: list of dictionaries, each stores annotation for one pod
    :param margin: bounding box margin added to original box
    :param shape: the shape (h, w) of the original image
    :returns : the updated annotation
    """
    anns = copy.deepcopy(anns)
    for id, ann in enumerate(anns):
        s = ann['keypoints']
        x = s[0::3]
        y = s[1::3]
        print('x:', x)
        print('y:', y)
        x0 = max(0, round(np.min([a for a in x if a != 0])) - margin)
        y0 = max(0, round(np.min([a for a in y if a != 0])) - margin)

        if not shape:
            x1 = round(np.max(x)) + margin
            y1 = round(np.max(y)) + margin
        else:
            x1 = min(round(np.max(x)) + margin, shape[1])
            y1 = min(round(np.max(y)) + margin, shape[0])

        ann['area'] = (x1 - x0) * (y1 - y0)
        ann['bbox'] = [x0, y0, x1-x0, y1-y0]
    return anns


def loadRes(res_file):
    """
    Load temporary result file
    :param res_file: the path of temporary result json file
    :returns : the regularized annotation
    """
    with open(res_file) as f:
        anns = json.load(f)
    assert type(anns) == list, 'results in not an array of objects'
    return add_bbox_info(anns)


def get_outputs(img, model, preprocess):
    """Computes the averaged heatmap and paf for the given image
    :param img: numpy array, the image being processed
    :param model: pytorch model
    :param preprocess: the preprocess method
    :returns: the averaged paf and heatmap and a scale param
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

    from evaluate.evaluation import NEW_OPENPOSE
    if NEW_OPENPOSE:
        # for new openpose model
        paf = output2[0].cpu().data.numpy().transpose(0, 2, 3, 1)[0]
        heatmap = output2[1].cpu().data.numpy().transpose(0, 2, 3, 1)[0]
    else:
        paf = output1.cpu().data.numpy().transpose(0, 2, 3, 1)[0]
        heatmap = output2.cpu().data.numpy().transpose(0, 2, 3, 1)[0]

    return paf, heatmap, im_scale


def append_result(image_name, pods, upsample_keypoints, outputs):
    """Build the outputs to be evaluated
    :param image_name: str, the filename of the current image
    :param pods: list of pod
    :param upsample_keypoints: tuple of keypoint position
    :param outputs: list of dictionaries with the following keys: image_name,
                    category_id, keypoints, score
    """
    for pod in pods:
        one_result = {"image_name": image_name,
                      "category_id": 1,
                      "keypoints": [],
                      "score": 0}
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

        one_result["score"] = pod.score
        one_result["keypoints"] = list(keypoints.reshape(15))
        outputs.append(one_result)


def run_eval(image_dir, vis_dir, model, preprocess):
    """Run the evaluation on the test set and report mAP score
    :param image_dir: the directory containing images
    :param vis_dir: directory to store output images
    :param model: the model to test
    :returns: float, the reported mAP score
    """
    img_paths, ann_paths = get_soybean_dataset(image_dir)
    print("Total number of validation images {}".format(len(img_paths)))

    # iterate all val images
    outputs = []
    print("Processing Images in validation set")
    for i in range(len(img_paths)):
        cur_outputs = []
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
        pods = paf_to_pods_cpp(heatmap, paf, cfg)
        print('pods:', pods)

        # subset indicated how many pods found in this image.
        upsample_keypoints = (
        heatmap.shape[0] * cfg.MODEL.DOWNSAMPLE / scale_img, heatmap.shape[1] * cfg.MODEL.DOWNSAMPLE / scale_img)
        append_result(os.path.basename(img_paths[i]), pods, upsample_keypoints, outputs)
        append_result(os.path.basename(img_paths[i]), pods, upsample_keypoints, cur_outputs)

        out = draw_pods(oriImg, pods)
        bbox = add_bbox_info(cur_outputs, shape=oriImg.shape[:2])
        out = draw_bbox(out, bbox)
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        vis_path = os.path.join(vis_dir, file_name)
        cv2.imwrite(vis_path, out)
    # Eval and show the final result!
    return eval_bean(outputs=outputs, ann_paths=ann_paths)




