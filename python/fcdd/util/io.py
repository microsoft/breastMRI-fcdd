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
from fcdd.util.safety import MAX_CONFIG_BYTES, bounded_reader, confined_path, canonical_root_path, validate_resources


def read_cfg(cfg_file: str):
    """ Reads a given configuration file from disk and transforms it into a json dictionary of parameters """
    with bounded_reader(cfg_file, MAX_CONFIG_BYTES, text=True) as text:
        start = text.find('{')
        if start < 0:
            raise ValueError('Configuration does not contain a JSON object')
        cfg, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a JSON object')
    return cfg


def extract_args(args: Namespace, cfg: dict):
    """ Extracts all parameters found in the cfg configuration dictionary and put them in the argparse Namespace """
    validate_resources(cfg)
    output_root = canonical_root_path(args.logdir)
    config_logdir = cfg['logdir']
    logdir = output_root if config_logdir == args.logdir else confined_path(output_root, config_logdir)
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
    args.logdir = logdir
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