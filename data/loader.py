import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.io import read_image
import json
import os
import cv2
import random
from PIL import Image


class VidHOI_keyframe_Dataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 keyframe_folder: str, 
                 annotation, 
                 frames_per_clip,
                 split ='train',  
                 transform=None):
        super(VidHOI_keyframe_Dataset, self).__init__()
        self.root_dir = root_dir  # /home/da/Desktop/Dataset/Vid_HOI/
        self.keyframe_folder = keyframe_folder  # keyframe_train/ or keyframe_val/
        self.video_folders = np.sort(os.listdir(os.path.join(self.root_dir, self.keyframe_folder)))  #  ['0000', '0001', ...]
        self.anno = annotation
        self.frames_per_clip = frames_per_clip
        self.split = split
        self.transform = transform
        self.all_clips = []

        for i in range(len(self.video_folders)):
            video_ids = np.sort(os.listdir(os.path.join(self.root_dir, self.keyframe_folder, 
                                    self.video_folders[i])))  # ['3065125389', '2794976541', ...]
            # print("video_ids: ", video_ids)
            for j in range(len(video_ids)):
                one_video_real_kframes = []
                keyframes = np.sort(os.listdir(os.path.join(self.root_dir, self.keyframe_folder, 
                            self.video_folders[i], video_ids[j])))
                            # keyframes in a video: ['2440175990_000045.jpg', '2440175990_000075.jpg', ...]
                for ii in range(len(keyframes)):
                    real_keyframes = os.path.join(self.video_folders[i], video_ids[j], keyframes[ii])
                    one_video_real_kframes.append(real_keyframes)
                assert len(keyframes) == len(one_video_real_kframes)
                # print("all_real_keyframes: ", self.all_real_keyframes)

                if len(keyframes) <= self.frames_per_clip:
                    self.all_clips.append(one_video_real_kframes)
                else:
                    num_clip = np.ceil(len(real_keyframes) / frames_per_clip)
                    for t in range(int(num_clip)):
                        if (t+1) * self.frames_per_clip <= len(real_keyframes):
                            clip = one_video_real_kframes[t * self.frames_per_clip : (t+1) * self.frames_per_clip]
                        else:
                            # clip = one_video_real_kframes[len(keyframes) - self.frames_per_clip : len(keyframes)]
                            clip = one_video_real_kframes[t * self.frames_per_clip : ]
                        self.all_clips.append(clip)
        print("num of all clips:", len(self.all_clips))  # train: 27087; val: 3216

        with open(self.anno, 'r') as f:
            self.label = json.load(f)
        # print(len(self.label))  # 1393976

    def __len__(self):
        return len(self.all_clips)   # num of splitted clips in the dataset

    def __getitem__(self, idx):
        '''
        idx: the index of clips in the whole dataset
        '''
        # if self.split == 'train':
        #     random_index = random.choice(range(0, len(self.all_clips) - idx))
        # else:
        #     random_index = 0

        kf_names = self.all_clips[idx]  # ['1027/9048441759/9048441759_000255.jpg', ...]
        # print(len(kf_names))  # 8
        kf_imgs = []  # transformed image for neural network
        original_imgs = [] # original RGB values of video frames
        original_img_size = [] # original shape of frames
        for i in range(len(kf_names)):
            img = read_image(os.path.join(self.root_dir, self.keyframe_folder, kf_names[i]))
            # print("img size: ", img.shape)
            original_imgs.append(img)
            original_img_size.append(img.shape)
            if self.transform:
                img = self.transform(transforms.ToPILImage()(img))
            kf_imgs.append(img)
        # print("kf_imgs: ", kf_imgs)
        # print("keyframe images:", kf_imgs[0].shape)  # torch.Size([3, 224, 224])
        # print("original_img_size: ", original_img_size) # [torch.Size([3, 360, 640]), ...]

        # kf_labels = self.all_clip_label[idx]
        one_clip_label = []
        for i in range(len(kf_names)):
            kf_name = kf_names[i]
            one_frame_in_clip_label = []
            for t in range(len(self.label)):
                if (kf_name[:10] == self.label[t]['video_id']  and kf_name[11:17] == self.label[t]['frame_id']):
                    # keyframe_name = label[t]['video_id'] + '_' + label[t]['frame_id'] + '.jpg'
                    # print("video_id: {}, frame_id: {}".format(self.label[t]['video_id'], self.label[t]['frame_id']))
                    frame_label = []
                    frame_label.append(self.label[t]['person_box']['xmin'])
                    frame_label.append(self.label[t]['person_box']['ymin'])
                    frame_label.append(self.label[t]['person_box']['xmax'])
                    frame_label.append(self.label[t]['person_box']['ymax'])
                    frame_label.append(self.label[t]['object_box']['xmin'])
                    frame_label.append(self.label[t]['object_box']['ymin'])
                    frame_label.append(self.label[t]['object_box']['xmax'])
                    frame_label.append(self.label[t]['object_box']['ymax'])
                    frame_label.append(self.label[t]['person_id'])
                    frame_label.append(self.label[t]['object_id'])
                    frame_label.append(self.label[t]['object_class'])
                    frame_label.append(self.label[t]['action_class'])
                    one_frame_in_clip_label.append(frame_label)
            # print(one_frame_in_clip_label)
            # print('-' * 50)
            # print(len(one_frame_in_clip_label))
            # one_clip_label['{}'.format(i)] = one_frame_in_clip_label
            one_clip_label.append(one_frame_in_clip_label)
        # print("len of one clip label: ", len(one_clip_label))
        # print(one_clip_label)

        # for i in range(len(one_clip_label)):
        #     print("number of one frame label: ", len(one_clip_label['{}'.format(i)]))
        # print("-" * 50)

        sample = {'clip_frames': kf_imgs,
                  'original_frames': original_imgs,
                  'original_frame_size': original_img_size,
                  'clip_labels': one_clip_label, 
                  'keyframe_names': kf_names}

        return sample