# Copyright (c) 2021 liznerski (original FCDD work)
# Copyright (c) 2025 Microsoft Corporation (FCDD for breast cancer detection)
# Licensed under the MIT License.

import json
import os
import os.path as pt
import re
import sys
import warnings
from argparse import Namespace
from typing import List

import torch
from fcdd.util.logging import Logger


def read_cfg(cfg_file: str):
    """ Reads a given configuration file from disk and transforms it into a json dictionary of parameters """
    with open(cfg_file) as reader:
        cfg = reader.readlines()
        cfg = ' '.join(cfg)
        re.DOTALL = True
        pttn = re.compile('\{(.*)\}')
        cfg = pttn.findall(cfg)
        assert len(cfg) == 1
        cfg = cfg[0]
        cfg = json.loads('{{{}}}'.format(cfg))
        return cfg


def extract_args(args: Namespace, cfg: dict):
    """ Extracts all parameters found in the cfg configuration dictionary and put them in the argparse Namespace """
    args.bias = cfg['bias']
    args.optimizer_type = cfg['optimizer_type']
    args.preproc = cfg['preproc']
    args.quantile = cfg['quantile']
    args.scheduler_type = cfg['scheduler_type']
    args.supervise_mode = cfg['supervise_mode']
    args.batch_size = cfg['batch_size']
    args.epochs = cfg['epochs']
    args.workers = cfg['workers']
    args.learning_rate = cfg['learning_rate']
    args.weight_decay = cfg['weight_decay']
    args.lr_sched_param = cfg['lr_sched_param']
    args.dataset = cfg['dataset']
    args.net = cfg['net']
    args.datadir = cfg['datadir']
    args.normal_class = cfg['normal_class']
    args.acc_batches = cfg['acc_batches']
    args.objective = cfg['objective']
    args.logdir = cfg['logdir']
    args.load = cfg['load']
    args.noise_mode = cfg['noise_mode']
    args.oe_limit = cfg['oe_limit']
    args.online_supervision = cfg['online_supervision']
    args.nominal_label = cfg['nominal_label']
    args.blur_heatmaps = cfg['blur_heatmaps']
    args.gauss_std = cfg['gauss_std']
    args.resdown = cfg['resdown']
    args.normal_class = cfg['normal_class']
    args.readme = ''
    args.cuda = True
    return args


OPTIONS = ['base', 'hsc', 'gts', 'bce']