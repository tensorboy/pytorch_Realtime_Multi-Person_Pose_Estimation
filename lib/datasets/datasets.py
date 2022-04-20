import copy
import json
import logging
import os
import numpy as np
import torch.utils.data
import torchvision
from PIL import Image

from .heatmap import putGaussianMaps
from .paf import putVecMaps
from . import transforms, utils



def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('neck'), keypoints.index('right_hip')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('neck'), keypoints.index('left_hip')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('neck'), keypoints.index('right_shoulder')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('right_shoulder'), keypoints.index('right_eye')],
        [keypoints.index('neck'), keypoints.index('left_shoulder')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_eye')],
        [keypoints.index('neck'), keypoints.index('nose')],
        [keypoints.index('nose'), keypoints.index('right_eye')],
        [keypoints.index('nose'), keypoints.index('left_eye')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')]
    ]  # len: 19
    return kp_lines


def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'neck',
        'right_shoulder',
        'right_elbow',
        'right_wrist',
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'right_hip',
        'right_knee',
        'right_ankle',
        'left_hip',
        'left_knee',
        'left_ankle',
        'right_eye',
        'left_eye',
        'right_ear',
        'left_ear']     # len: 18

    return keypoints


def get_soybean_keypoints():
    """Get the soybean keypoints."""
    keypoints = [
        'first_bean',
        'second_bean',
        'third_bean',
        'fourth_bean',
        'fifth_bean'
    ]      # len: 5
    return keypoints



def kp_soybean_connections(keypoints):
    kp_lines = [
        [keypoints.index('first_bean'), keypoints.index('second_bean')],
        [keypoints.index('second_bean'), keypoints.index('third_bean')],
        [keypoints.index('third_bean'), keypoints.index('fourth_bean')],
        [keypoints.index('fourth_bean'), keypoints.index('fifth_bean')]
    ]   # len: 4
    return kp_lines


def collate_images_anns_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    anns = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return images, anns, metas


def collate_multiscale_images_anns_meta(batch):
    """Collate for multiscale.

    indices:
        images: [scale, batch , ...]
        anns: [batch, scale, ...]
        metas: [batch, scale, ...]
    """
    n_scales = len(batch[0][0])
    images = [torch.utils.data.dataloader.default_collate([b[0][i] for b in batch])
              for i in range(n_scales)]
    anns = [[b[1][i] for b in batch] for i in range(n_scales)]
    metas = [b[2] for b in batch]
    return images, anns, metas


def collate_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    targets1 = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    targets2 = torch.utils.data.dataloader.default_collate([b[2] for b in batch])

    return images, targets1, targets2


def get_soybean_dataset(data_dir):
    # get all img paths and anns paths
    imgs, anns = [], []
    for root, dir, files in os.walk(data_dir):
        files = sorted(files)
        i = 0
        while i < len(files):
            if i + 1 == len(files):
                i += 1
                continue
            name_img, ext_img = os.path.splitext(files[i])
            name_ann, ext_ann = os.path.splitext(files[i + 1])
            # print(name_img, name_ann)

            if ext_img == '.jpg' and ext_ann == '.json' and name_img == name_ann:
                imgs.append(os.path.join(root, files[i]))
                anns.append(os.path.join(root, files[i + 1]))
            i += 2

    return imgs, anns


class CocoKeypoints(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Based on `torchvision.dataset.CocoDetection`.

    Caches preprocessing.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, image_transform=None, target_transforms=None,
                 n_images=None, preprocess=None, all_images=False, all_persons=False, input_y=368, input_x=368,
                 stride=8):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)

        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        if all_images:
            self.ids = self.coco.getImgIds()
        elif all_persons:
            self.ids = self.coco.getImgIds(catIds=self.cat_ids)
        else:
            self.ids = self.coco.getImgIds(catIds=self.cat_ids)
            self.filter_for_keypoint_annotations()
        if n_images:
            self.ids = self.ids[:n_images]
        print('Images: {}'.format(len(self.ids)))

        self.preprocess = preprocess or transforms.Normalize()
        self.image_transform = image_transform or transforms.image_transform
        self.target_transforms = target_transforms

        self.HEATMAP_COUNT = len(get_keypoints())
        self.LIMB_IDS = kp_connections(get_keypoints())
        self.input_y = input_y
        self.input_x = input_x
        self.stride = stride
        self.log = logging.getLogger(self.__class__.__name__)

    def filter_for_keypoint_annotations(self):
        print('filter for keypoint annotations ...')

        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):  # Only keeps instance with keypoint annotation.
                    return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_keypoint_annotation(image_id)]
        print('... done.')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)
        anns = copy.deepcopy(anns)

        image_info = self.coco.loadImgs(image_id)[0]
        self.log.debug(image_info)
        # PIL image form
        with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            # original images of different sizes, ex: (480, 640), (500, 375), (501, 640)
            image = Image.open(f).convert('RGB')

        # What is this ?
        meta_init = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
        }

        image, anns, meta = self.preprocess(image, anns, None)  # preprocessing -> image: (368, 368)

        if isinstance(image, list):
            return self.multi_image_processing(image, anns, meta, meta_init)

        return self.single_image_processing(image, anns, meta, meta_init)  # do processing -> image: torch [3, 368, 368]

    def multi_image_processing(self, image_list, anns_list, meta_list, meta_init):
        return list(zip(*[
            self.single_image_processing(image, anns, meta, meta_init)
            for image, anns, meta in zip(image_list, anns_list, meta_list)
        ]))

    def single_image_processing(self, image, anns, meta, meta_init):
        meta.update(meta_init)

        # transform image
        original_size = image.size  # (368, 368)
        image = self.image_transform(image)  # do some transform -> [3, 368, 368]
        assert image.size(2) == original_size[0]
        assert image.size(1) == original_size[1]

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        self.log.debug(meta)

        heatmaps, pafs = self.get_ground_truth(anns)

        heatmaps = torch.from_numpy(
            heatmaps.transpose((2, 0, 1)).astype(np.float32))  # [19, 46, 46]

        pafs = torch.from_numpy(pafs.transpose((2, 0, 1)).astype(np.float32))  # [38, 46, 46]
        return image, heatmaps, pafs  # [3, 368, 368], [19, 46, 46], [38, 46, 46]

    def remove_illegal_joint(self, keypoints):
        # keypoints shape: (x, 18, 3), x is the number of keypoints in the image
        MAGIC_CONSTANT = (-1, -1, 0)
        mask = np.logical_or.reduce((keypoints[:, :, 0] >= self.input_x,
                                     keypoints[:, :, 0] < 0,
                                     keypoints[:, :, 1] >= self.input_y,
                                     keypoints[:, :, 1] < 0))
        keypoints[mask] = MAGIC_CONSTANT

        return keypoints

    def add_neck(self, keypoint):
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
        right_shoulder = keypoint[6, :]
        left_shoulder = keypoint[5, :]
        neck = (right_shoulder + left_shoulder) / 2
        if right_shoulder[2] == 2 and left_shoulder[2] == 2:
            neck[2] = 2
        else:
            neck[2] = right_shoulder[2] * left_shoulder[2]

        neck = neck.reshape(1, len(neck))
        neck = np.round(neck)
        keypoint = np.vstack((keypoint, neck))
        keypoint = keypoint[our_order, :]

        return keypoint

    def get_ground_truth(self, anns):

        grid_y = int(self.input_y / self.stride)
        grid_x = int(self.input_x / self.stride)
        channels_heat = (self.HEATMAP_COUNT + 1)
        channels_paf = 2 * len(self.LIMB_IDS)
        heatmaps = np.zeros((int(grid_y), int(grid_x), channels_heat))
        pafs = np.zeros((int(grid_y), int(grid_x), channels_paf))

        keypoints = []
        # for each person in the image
        for ann in anns:
            single_keypoints = np.array(ann['keypoints']).reshape(17, 3)
            single_keypoints = self.add_neck(single_keypoints)
            keypoints.append(single_keypoints)
        keypoints = np.array(keypoints)
        keypoints = self.remove_illegal_joint(keypoints)

        # confidance maps for body parts
        for i in range(self.HEATMAP_COUNT):
            joints = [jo[i] for jo in keypoints]
            for joint in joints:
                if joint[2] > 0.5:
                    center = joint[:2]
                    gaussian_map = heatmaps[:, :, i]
                    heatmaps[:, :, i] = putGaussianMaps(
                        center, gaussian_map,
                        7.0, grid_y, grid_x, self.stride)
        # pafs
        for i, (k1, k2) in enumerate(self.LIMB_IDS):
            # limb
            count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
            for joint in keypoints:
                if joint[k1, 2] > 0.5 and joint[k2, 2] > 0.5:
                    centerA = joint[k1, :2]
                    centerB = joint[k2, :2]
                    vec_map = pafs[:, :, 2 * i:2 * (i + 1)]

                    pafs[:, :, 2 * i:2 * (i + 1)], count = putVecMaps(
                        centerA=centerA,
                        centerB=centerB,
                        accumulate_vec_map=vec_map,
                        count=count, grid_y=grid_y, grid_x=grid_x, stride=self.stride
                    )


        # background
        heatmaps[:, :, -1] = np.maximum(
            1 - np.max(heatmaps[:, :, :self.HEATMAP_COUNT], axis=2),
            0.
        )
        return heatmaps, pafs

    def __len__(self):
        return len(self.ids)


class SoybeanKeypoints(torch.utils.data.Dataset):
    def __init__(self, root, stride=8, image_transform=None,
                 target_transforms=None, preprocess=None, input_y=368, input_x=368):
        self.root = root
        self.imgs = []
        self.anns = []
        self.preprocess = preprocess or transforms.NormalizeBean()
        self.image_transform = image_transform or transforms.image_transform
        self.target_transforms = target_transforms
        self.input_y = input_y
        self.input_x = input_x
        self.stride = stride

        self.MAX_BEAN_COUNT = len(get_soybean_keypoints())
        self.BEAN_CONNECTION_IDS = kp_soybean_connections(get_soybean_keypoints())
        self.log = logging.getLogger(self.__class__.__name__)

        # get all img paths and anns paths
        self.imgs, self.anns = get_soybean_dataset(self.root)
        print('total images:', len(self.imgs))
        print('total annotations:', len(self.anns))

    def __getitem__(self, index):
        ann_path = self.anns[index]
        anns = self.loadAnns(ann_path)  # load annotation of one image
        anns = copy.deepcopy(anns)

        with open(os.path.join(self.root, self.imgs[index]), 'rb') as f:
            image = Image.open(f).convert('RGB')

        # What is this ?
        meta_init = {
            'image_path': anns['imagePath'],
        }

        image, anns, meta = self.preprocess(image, anns, None)
        return self.bean_image_processing(image, anns, meta, meta_init)

    def __len__(self):
        return len(self.anns)

    def loadAnns(self, ann_path):
        f = open(ann_path)
        anns = json.load(f)
        return anns

    def get_ground_truth(self, anns):
        grid_y = int(self.input_y / self.stride)
        grid_x = int(self.input_x / self.stride)
        channels_heat = (self.MAX_BEAN_COUNT + 1)           #  plus 1? because of background
        channels_paf = 2 * len(self.BEAN_CONNECTION_IDS)
        heatmaps = np.zeros((int(grid_y), int(grid_x), channels_heat))
        pafs = np.zeros((int(grid_y), int(grid_x), channels_paf))

        keypoints = []
        for ann in anns['annotations']:
            single_keypoints = ann['keypoints']
            if len(single_keypoints) < 5:
                single_keypoints = np.concatenate((single_keypoints, np.array([[0, 0, 0]] * (5-len(single_keypoints)))))
            keypoints.append(single_keypoints)
        keypoints = np.array(keypoints)
        keypoints = self.remove_illegal_joint(keypoints)

        # confidence maps for beans
        for i in range(self.MAX_BEAN_COUNT):
            beans = [jo[i] for jo in keypoints]
            for center in beans:
                if center[2] > 0.5:
                    gaussian_map = heatmaps[:, :, i]
                    heatmaps[:, :, i] = putGaussianMaps(
                        center, gaussian_map,
                        7.0, grid_y, grid_x, self.stride)

        # pafs
        for i, (k1, k2) in enumerate(self.BEAN_CONNECTION_IDS):
            # limb
            count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
            for pod in keypoints:
                if pod[k1, 2] > 0.5 and pod[k2, 2] > 0.5:
                    centerA = pod[k1, :2]
                    centerB = pod[k2, :2]

                    vec_map = pafs[:, :, 2 * i:2 * (i + 1)]
                    pafs[:, :, 2 * i:2 * (i + 1)], count = putVecMaps(
                        centerA=centerA,
                        centerB=centerB,
                        accumulate_vec_map=vec_map,
                        count=count, grid_y=grid_y, grid_x=grid_x, stride=self.stride
                    )
        return heatmaps, pafs

    def remove_illegal_joint(self, keypoints):
        # keypoints shape: (x, 5, 3), x is the number of keypoints in the image
        MAGIC_CONSTANT = (-1, -1, 0)    # replace illegal point location to MAGIC_CONSTANT
        mask = np.logical_or.reduce((keypoints[:, :, 0] >= self.input_x,
                                     keypoints[:, :, 0] < 0,
                                     keypoints[:, :, 1] >= self.input_y,
                                     keypoints[:, :, 1] < 0))
        keypoints[mask] = MAGIC_CONSTANT
        return keypoints

    def bean_image_processing(self, image, anns, meta, meta_init):
        meta.update(meta_init)

        # transform image
        original_size = image.size  # (368, 368)
        image = self.image_transform(image)  # do some transform -> [3, 368, 368]
        assert image.size(2) == original_size[0]
        assert image.size(1) == original_size[1]

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        self.log.debug(meta)

        heatmaps, pafs = self.get_ground_truth(anns)

        # [6, 46, 46] (6 = 5 bean max + 1 background)
        heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1)).astype(np.float32))

        # [8, 46, 46]  (8 = 4 connections * 2 vector dim)
        pafs = torch.from_numpy(pafs.transpose((2, 0, 1)).astype(np.float32))

        return image, heatmaps, pafs


class ImageList(torch.utils.data.Dataset):
    def __init__(self, image_paths, preprocess=None, image_transform=None):
        self.image_paths = image_paths
        self.image_transform = image_transform or transforms.image_transform
        self.preprocess = preprocess

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.preprocess is not None:
            image = self.preprocess(image, [], None)[0]

        original_image = torchvision.transforms.functional.to_tensor(image)
        image = self.image_transform(image)

        return image_path, original_image, image

    def __len__(self):
        return len(self.image_paths)


class PilImageList(torch.utils.data.Dataset):
    def __init__(self, images, image_transform=None):
        self.images = images
        self.image_transform = image_transform or transforms.image_transform

    def __getitem__(self, index):
        pil_image = self.images[index].copy().convert('RGB')
        original_image = torchvision.transforms.functional.to_tensor(pil_image)
        image = self.image_transform(pil_image)

        return index, original_image, image

    def __len__(self):
        return len(self.images)
