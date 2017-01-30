#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

import chainer
from chainer import iterators
from chainer import training
from chainer.training import extensions
from lib.dataset_custom import DatasetObjectDetection
from updater import ParallelUpdater
from utils.prepare_train import create_args
from utils.prepare_train import create_logger
from utils.prepare_train import create_result_dir
from utils.prepare_train import get_model
from utils.prepare_train import get_optimizer

import logging

args = create_args()
# result_dir = create_result_dir(args.model_name)
# create_logger(args, result_dir)

# Prepare devices
gpu_id = int(args.gpus)#currently only one gpu. -1 or 0 

# Instantiate a model
model = get_model(
    args.model_file, args.model_name, gpu_id, args.rpn_in_ch,
    args.rpn_out_ch, args.n_anchors, args.feat_stride, args.anchor_scales,
    args.num_classes, args.spatial_scale, args.rpn_sigma, args.sigma,
    args.trunk_file, args.trunk_name, args.trunk_param, True, result_dir=None)

#To GPU
if gpu_id >= 0:
    model.to_gpu()

# Instantiate a optimizer
optimizer = get_optimizer(
    model, args.opt, args.lr, args.adam_alpha, args.adam_beta1,
    args.adam_beta2, args.adam_eps, args.weight_decay)

# Setting up datasets
dataset = DatasetObjectDetection(args.train_img_dir, "./data/voc_train.json")
valid = DatasetObjectDetection(args.train_img_dir, "./data/voc_val.json")


batch_size=1

current_epoch=dataset.epoch

while (dataset.epoch <= args.epoch):
    img, im_info, gt_boxes = dataset.get_batch(batch_size)
    loss = model(img, im_info, gt_boxes)
    #rpn_cls_loss, rpn_loss_bbox, loss_bbox, loss_cls = loss
    for i in xrange(4):
        optimizer.zero_grads()
        loss[i].backward()
        optimizer.update()
        print(loss[i].data)
    print("--")