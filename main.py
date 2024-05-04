# -*- coding:utf-8 -*-
import os.path as osp
import argparse
import os
import random
import json
import copy
from collections import defaultdict
import numpy as np
import torch
import time
from tools.config import Config
from tools.utils import mkdir_or_exist, set_random_seed
from tools.train import train_model, evaluate
from tools.logging_ import get_root_logger
from models.builder import build_classifier
from datasets.builder import build_dataset
import copy
import pickle
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler,
                              TensorDataset)



CUDA_LAUNCH_BLOCKING = 1


def parse_args():
    parser = argparse.ArgumentParser(description='Train a ZeroDDI')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')

    parser.add_argument(
        '--device', default="cuda:0", help='cuda:0 or cuda:1')
    parser.add_argument(
        '--seednumber', default=42, help='number of seeds')

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--deterministic',
        default=True,
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
    #parser.add_argument('--test', type=str, default='no', help='zsl or gzsl or no')
    parser.add_argument('--zsl_para', type=str, default=False)
    parser.add_argument('--gzsl_para', type=str, default=False)

    args = parser.parse_args()

    # add args.local_rank into os.environ
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def unify_seed_device(cfg, i, det, dev,args):
    """
    Unify all seeds and devices in a work.
    """
    set_random_seed(i, deterministic=det)

    cfg.device = dev
    cfg.model.device = dev
    cfg.model.leftmodel.device = dev
    cfg.model.rightmodel.device = dev
    cfg.data.train.device = dev
    #if not args.case:
    cfg.data.zsl_test.device = dev
    cfg.data.zsl_val.device = dev
    cfg.data.gzsl_test.device = dev
    cfg.data.gzsl_val.device = dev
    cfg.data.val_seen.device = dev
    cfg.data.test_seen.device = dev


def main():
    args = parse_args()
    # get config from file
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)



    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    mkdir_or_exist(osp.join(cfg.work_dir, 'model_parameter'))
    cfg.model_parameter_epoch = osp.join(osp.join(cfg.work_dir, 'model_parameter'),
                                         f'model_epoch{cfg.num_epochs}_seed{args.seednumber}.pkl')
    cfg.model_parameter_best = osp.join(osp.join(cfg.work_dir, 'model_parameter'),
                                        f'model_best_epoch{cfg.num_epochs}_seen{args.seednumber}.pkl')
    # timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.zsl_para or args.gzsl_para:
        log_file = osp.join(cfg.work_dir, f'eval_{timestamp}.log')
    else:
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')

    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    logger.info(f"config is {cfg}")
    #logger.info(f'Set random seed to {int(args.seednumber)}')

    # set random seeds
    # pytorch has seed, cuda has seed
    unify_seed_device(cfg, int(args.seednumber), args.deterministic, cfg.device,args)
    cfg.seednumber = int(args.seednumber)

    if args.zsl_para or args.gzsl_para:  # test

        train_dataset = build_dataset(cfg.data.train)
        cfg.model.rightmodel.input_dim = train_dataset.input_dim
        cfg.model.seen_labels = train_dataset.current_dataset_eventid_uni
        zsl_test_dataset = build_dataset(cfg.data.zsl_test)
        gzsl_test_dataset = build_dataset(cfg.data.gzsl_test)
        seen_test_dataset = build_dataset(cfg.data.test_seen)
        cfg.model.rightmodel.output_dim = zsl_test_dataset.dim
        cfg.model.zsl_labels = zsl_test_dataset.current_dataset_eventid_uni
        cfg.model.gzsl_labels = gzsl_test_dataset.current_dataset_eventid_uni

        cfg.model.train_rightinput = seen_test_dataset.rightinput
        cfg.model.val_zsl_rightinput = zsl_test_dataset.rightinput
        cfg.model.val_gzsl_rightinput = gzsl_test_dataset.rightinput

        model = build_classifier(cfg.model)
        model2 = build_classifier(cfg.model)
        model.to(cfg.device)
        model2.to(cfg.device)

        print(args.zsl_para)
        #model.load_state_dict(torch.load(args.zsl_para))
        #model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.zsl_para,map_location=cfg.device).items()})
        model.load_state_dict(torch.load(args.zsl_para))
        print(args.gzsl_para)
        model2.load_state_dict(torch.load(args.gzsl_para))

        # seen_labels = test_dataset.seen_labels
        acc = evaluate(model, zsl_test_dataset, logger, cfg, "zsl", visualize_acc=True)
        H = evaluate(model2, gzsl_test_dataset, logger, cfg, "gzsl", visualize_acc=True)

    else:  # train
        print("train")
        datasets = []
        train_dataset = build_dataset(cfg.data.train)
        cfg.model.rightmodel.input_dim = train_dataset.input_dim
        cfg.model.rightmodel.output_dim = train_dataset.dim
        cfg.model.seen_labels = train_dataset.current_dataset_eventid_uni
        zsl_val_dataset = build_dataset(cfg.data.zsl_val)
        gzsl_val_dataset = build_dataset(cfg.data.gzsl_val)
        cfg.model.zsl_labels = zsl_val_dataset.current_dataset_eventid_uni
        cfg.model.gzsl_labels = gzsl_val_dataset.current_dataset_eventid_uni

        datasets.append(train_dataset)
        datasets.append(zsl_val_dataset)
        datasets.append(gzsl_val_dataset)

        cfg.model.train_rightinput = train_dataset.rightinput
        cfg.model.val_zsl_rightinput = zsl_val_dataset.rightinput
        cfg.model.val_gzsl_rightinput = gzsl_val_dataset.rightinput
        cfg.model.attributlabel = train_dataset.rightattributelabel
        # create models
        model = build_classifier(cfg.model)
        print("cfg.device",cfg.device)
        model.to(cfg.device)
        logger.info(f"model is {model}")


        print("begin to train model")
        train_model(model, datasets, cfg)


if __name__ == "__main__":
    main()