import numpy as np
import cv2


def resize(frame, desired_size):
    old_size = frame.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    frame = cv2.resize(frame, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    frame = cv2.copyMakeBorder(
        frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    if left == 0:
        scale = float(old_size[1]) / desired_size
    else:
        scale = float(old_size[0]) / desired_size
    return frame, left, top, scale


def imcv2_recolor(im, a=.1):
    # t = [np.random.uniform()]
    # t += [np.random.uniform()]
    # t += [np.random.uniform()]
    # t = np.array(t) * 2. - 1.
    t = np.random.uniform(-1, 1, 3)

    # random amplify each channel
    im = im.astype(np.float)
    im *= (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform(-1, 1)
    im = np.power(im / mx, 1. + up * .5)
    # return np.array(im * 255., np.uint8)
    return im


def imcv2_affine_trans(im, flip=None, im_shape=None, rotate=False, max_scale=1.5):
    # Scale and translate
    h, w = im.shape[:2] if im_shape is None else im_shape[:2]
    scale = np.random.uniform(1., max_scale)

    degree = np.random.uniform(-5, 5) if rotate else None

    max_offx = (scale - 1.) * w
    max_offy = (scale - 1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)

    flip_ = np.random.uniform() > 0.5 if flip is None else flip

    if im is not None:
        im = apply_affine(im, scale, [offx, offy, degree], flip_, im_shape)

    return im, [scale, [offx, offy, degree], flip_, im_shape]


def apply_affine(im, scale, offs, flip, im_shape=None):
    offx, offy, degree = offs
    h, w = im.shape[:2] if im_shape is None else im_shape[:2]

    im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
    if degree is not None:
        retval = cv2.getRotationMatrix2D((w // 2, h // 2), degree, 1)
        im = cv2.warpAffine(im, retval, (w, h))
    im = im[offy: (offy + h), offx: (offx + w)]
    if flip:
        im = cv2.flip(im, 1)

    return im


def offset_boxes(boxes, scale, offs, flip, im_shape):
    if len(boxes) == 0:
        return boxes

    boxes = np.asarray(boxes, dtype=np.float)
    expand = False
    if boxes.ndim == 1:
        expand = True
        boxes = np.expand_dims(boxes, 0)

    boxes *= scale
    boxes[:, 0::2] -= offs[0]
    boxes[:, 1::2] -= offs[1]

    is_box = boxes.shape[-1] % 4 == 0
    # if is_box:
    #     boxes = clip_boxes(boxes, im_shape)

    if flip:
        boxes[:, 0::2] = im_shape[1] - boxes[:, 0::2]
        if is_box:
            for i in range(boxes.shape[-1] // 4):
                tmp = boxes[:, i].copy()
                boxes[:, i] = boxes[:, i + 2]
                boxes[:, i + 2] = tmp

        # boxes_x = np.copy(boxes[:, 0])
        # boxes[:, 0] = im_shape[1] - boxes[:, 2]
        # boxes[:, 2] = im_shape[1] - boxes_x

    if expand:
        boxes = boxes[0]
    return boxes


def _factor_closest(num, factor, is_ceil=True):
    num = np.ceil(float(num) / factor) if is_ceil else np.floor(float(num) / factor)
    num = int(num) * factor
    return num


def crop_with_factor(im, dest_size=None, factor=32, is_ceil=True):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # im_scale = 1.
    # if max_size is not None and im_size_min > max_size:
    im_scale = float(dest_size) / im_size_min
    im = cv2.resize(im, None, fx=im_scale, fy=im_scale)

    h, w, c = im.shape
    new_h = _factor_closest(h, factor=factor, is_ceil=is_ceil)
    new_w = _factor_closest(w, factor=factor, is_ceil=is_ceil)
    im_croped = np.zeros([new_h, new_w, c], dtype=im.dtype)
    im_croped[0:h, 0:w, :] = im

    return im_croped, im_scale, im.shape
    
