"""MSCOCO Dataloader
   Thanks to @tensorboy @shuangliu
"""

try:
    import ujson as json
except ImportError:
    import json

from torchvision.transforms import ToTensor
from training.datasets.coco_data.COCO_data_pipeline import Cocokeypoints
from training.datasets.dataloader import sDataLoader


def get_loader(json_path, data_dir, mask_dir, inp_size, feat_stride, preprocess,
               batch_size, params_transform, training=True, shuffle=True, num_workers=3):
    """ Build a COCO dataloader
    :param json_path: string, path to jso file
    :param datadir: string, path to coco data
    :returns : the data_loader
    """
    with open(json_path) as data_file:
        data_this = json.load(data_file)
        data = data_this['root']

    num_samples = len(data)
    train_indexes = []
    val_indexes = []
    for count in range(num_samples):
        if data[count]['isValidation'] != 0.:
            val_indexes.append(count)
        else:
            train_indexes.append(count)

    coco_data = Cocokeypoints(root=data_dir, mask_dir=mask_dir,
                              index_list=train_indexes if training else val_indexes,
                              data=data, inp_size=inp_size, feat_stride=feat_stride,
                              preprocess=preprocess, transform=ToTensor(), params_transform=params_transform)

    data_loader = sDataLoader(coco_data, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers)

    return data_loader
