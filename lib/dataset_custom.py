#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import cv2 as cv
import glob
import numpy as np
import os
import json

class DatasetObjectDetection(chainer.dataset.DatasetMixin):

    LABELS = ('__background__',  # always index 0
              'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')
    IMG_TARGET_SIZE = 600
    IMG_MAX_SIZE = 1000

    def __init__(self, img_dir,json_file):
        self.mean = np.array([[[103.939, 116.779, 123.68]]])  # BGR VGG mean
        self.img_dir = img_dir
        self.objects=json.load(open(json_file))

    def __len__(self):
        return len(self.objects)

    def get_example(self, i):
        obj = self.objects[i]
        bbox = obj['bndbox']
        name = obj['name']
        clsid = self.LABELS.index(name)
        gt_boxes = np.asarray([bbox[0], bbox[1], bbox[2], bbox[3], clsid],
                              dtype=np.float32)

        # Load image
        img_fn = '{}/{}'.format(self.img_dir, obj['filename'])
        img = cv.imread(img_fn).astype(np.float)
        img -= self.mean

        # Scaling
        im_size_min = np.min(img.shape[:2])
        im_size_max = np.max(img.shape[:2])
        im_scale = float(self.IMG_TARGET_SIZE) / float(im_size_min)

        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > self.IMG_MAX_SIZE:
            im_scale = float(self.IMG_MAX_SIZE) / float(im_size_max)
        img = cv.resize(img, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv.INTER_LINEAR)
        h, w = img.shape[:2]
        im_info = np.asarray([h, w, im_scale], dtype=np.float32)
        img = img.transpose(2, 0, 1).astype(np.float32)

        return img, im_info, gt_boxes


if __name__ == '__main__':
    dataset = DatasetObjectDetection("../data/VOCdevkit/VOC2007/JPEGImages/","../data/voc_train.json")
    img, im_info, gt_boxes = dataset[0]
    print(img.shape)
    print(im_info)
    print(gt_boxes)
    print('len:', len(dataset))

    dataset = DatasetObjectDetection("../data/VOCdevkit/VOC2007/JPEGImages/","../data/voc_train.json")
    img, im_info, gt_boxes = dataset[0]
    print(img.shape)
    print(im_info)
    print(gt_boxes)
    print('len:', len(dataset))
