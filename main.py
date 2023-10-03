import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.roi_align
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from timm.models.layers import trunc_normal_
from torchvision.models.detection import (fasterrcnn_resnet50_fpn, 
                                          fasterrcnn_resnet50_fpn_v2, 
                                          FasterRCNN_ResNet50_FPN_Weights,
                                          FasterRCNN_ResNet50_FPN_V2_Weights)

import numpy as np
import os, tqdm
import gc
import time
import random
import argparse
from datetime import datetime
from models.args import get_args
import utils.factory as utils
from datasets.VidHOI_dataset import VidHOI_keyframe_Dataset
from models.ACoLP import ACoLP

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std = .02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def main():
    args = get_args()
    args.world_size = args.gpus * args.nodes
    device = torch.device('cuda', args.local_rank)
    BATCH_SIZE = args.batch_size
    # device = torch.device(args.device)
    exp_time = datetime.now().strftime("%Y-%m-%d_%H:%M")

    if args.local_rank == 0:
        print("#" * 80)
        # print("# - Experiment: {}".format(exp_descp))
        print("# - Experiment start on: {}".format(exp_time))
        print("# - {}".format(args))
        print("#" * 80)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    gc.collect()  # empty RAM
    torch.cuda.empty_cache()
    torch.cuda.set_device(args.local_rank)

    ROOT_PATH = '/home/data/Dataset/Vid_HOI'
    JSON_ANNO_TRAIN = '/home/data/Dataset/Vid_HOI/train_frame_annots.json'
    JSON_ANNO_VAL = '/home/data/Dataset/VidHOI_STHOI_paper_code/slowfast/datasets/vidor-github/val_frame_annots.json'

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.DEFAULT

    train_dst = VidHOI_keyframe_Dataset(
        root_dir = ROOT_PATH,
        keyframe_folder = 'small_keyframes_train',
        annotation = JSON_ANNO_TRAIN,
        frames_per_clip = 8, 
        transform=transforms.Compose(
            [transforms.Resize(args.re_size),
            # transforms.RandomHorizontalFlip(p=0.5), 
            # transforms.RandomCrop((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        ))

    val_dst = VidHOI_keyframe_Dataset(
        root_dir = ROOT_PATH,
        keyframe_folder = 'small_keyframes_test',
        annotation = JSON_ANNO_VAL,
        frames_per_clip = 8, 
        transform=transforms.Compose(
            [transforms.Resize(args.re_size),
            # transforms.RandomCrop((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        ))

    if args.local_rank == 0:
        print("Train size:", len(train_dst))  # Train size: 27087
        print("Val size:", len(val_dst))  # Val size: 3216

    # model = LangSupVidHOI()
    model = ACoLP()
    if args.distri:
        model.apply(init_weights)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model.to(device), device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        train_sampler = DistributedSampler(train_dst, num_replicas=None, rank=None, shuffle=True, seed=42, drop_last=False)
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, batch_size=BATCH_SIZE // 6, drop_last=True)
        train_dataloader = DataLoader(
            train_dst, num_workers=4, batch_sampler=train_batch_sampler)
    else:
        # model.apply(init_weights).to(device)
        model.apply(init_weights).cuda()
        train_dataloader = DataLoader(train_dst, batch_size=BATCH_SIZE // 6, num_workers=4,
                              shuffle=True, drop_last=False, collate_fn=lambda x: x)
        print("length of dataloader: ", len(train_dataloader)) # len(train_dst) / batch_size

    # for data in tqdm(total_dataloader, position=0, decs = 'load train dataset'):
    #     pass
    # total_dataloader.dataset.set_use_cache(use_cache=True)

    criterion = nn.BCEWithLogitsLoss().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr * args.lr_scale)
    # exp_lr_scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.9)

    for epoch in range(0, args.epochs):
        torch.cuda.empty_cache()
        total_step = len(train_dataloader)
        if args.local_rank == 0:
            print("Epoch:{}".format(epoch))
            print("total step: ", total_step)
        # train_sampler.set_epoch(epoch)
        epoch_since = time.time()
        # hs = open(os.path.join(save_path, "output.txt"), "a")

        # if args.local_rank == 0:
        #     hs.write("Mode: {} \t".format(args.mode))

        cunt_train = 0
        for i, sample_batched in enumerate(train_dataloader):
            cunt_train += 1
            # print(sample_batched[0]['clip_labels'])
            # inputs, labels = torch.tensor(sample_batched[0]['clip_frames']).to(device), torch.tensor(sample_batched[0]['clip_labels']).to(device)
            input_list = []
            ori_img_list = []
            bbox_ratio_list = []
            label_list = []
            frame_name_list = []
            for ii in range(len(sample_batched[0]['clip_frames'])):
                # print("sampled batch: ", sample_batched[0]['keyframe_names'])
                inputs = sample_batched[0]['clip_frames'][ii].requires_grad_(True).to(device)
                input_list.append(inputs)

                ori_clip_frames = sample_batched[0]['original_frames'][ii].to(device)
                ori_img_list.append(ori_clip_frames)

                ori_img_size = sample_batched[0]['original_frame_size'][ii]
                bbox_ratio_x = args.re_size[0] / ori_img_size[1]
                bbox_ratio_y = args.re_size[1] / ori_img_size[2]
                bbox_ratio_list.append((bbox_ratio_x, bbox_ratio_y))

                labels = torch.tensor(sample_batched[0]['clip_labels'][ii]).to(device)
                label_list.append(labels)

                frame_name = sample_batched[0]['keyframe_names'][ii]
                frame_name_list.append(frame_name)
                # print("inputs shape: ", inputs.shape)  # torch.Size([3, 224, 224])
                # edge_fea = model(inputs, labels)
                print("frame: ", frame_name)
                torch.cuda.synchronize()
            print("-" * 50)
            print("cunt_train: ", cunt_train)
            print("bbox ratio: ", bbox_ratio_list)
            # print("input_list: ", input_list)
            # print("HOI label_list: ", label_list)

            # fea = model(input_list)
            fea = model(ori_img_list, input_list, frame_name_list, bbox_ratio_list)

if __name__=='__main__':
    main()