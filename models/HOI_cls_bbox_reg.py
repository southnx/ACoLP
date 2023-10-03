"""
HOI classification and bbox regression with action prompts and HOI prompts
"""
import os
from os.path import join
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import CosineSimilarity
import numpy as np
import json
from torchvision.ops import box_iou

# from .mlp import MLP
import mlp

def remove(string):
    return string.replace(" ", "")

def remove2(string):
    return string.replace("\n", "")

actions = [
            'next_to', 'in_front_of', 'behind', 'watch', 'lean_on', 'above', 'away', 'towards', 'hold', 'play_instrument', 
            'ride', 'hug', 'speak_to', 'inside', 'hold_hand_of', 'beneath', 'carry', 'use', 'caress', 'push', 'touch', 
            'press', 'wave', 'feed', 'pull', 'bite', 'hit', 'lift', 'pat', 'drive', 'grab', 'kiss', 'chase', 'release', 
            'point_to', 'clean', 'wave_hand_to', 'squeeze', 'kick', 'cut', 'shout_at', 'get_on', 'throw', 'shake_hand_with', 
            'get_off', 'open', 'smell', 'close', 'knock', 'lick'
        ]
# print(len(actions)) # 50

HOIs = []
hoi_comb_file = '/home/da/Desktop/Code/Vid_HOI/utils/triplet_cls.txt'
with open(hoi_comb_file) as f:
    lines = f.readlines()
    for i in range(len(lines)):
        l = remove(lines[i])
        l = remove2(l)
        # newHOI = l.split(",")
        # print(newHOI[1])
        HOIs.append(l)
# # print(HOIs[0][1])
# print(HOIs[100])
# string = "person,next_to,bird"
# print(string == HOIs[100])

HOI_actions = [
    'next_to', 'towards', 'point_to', 'in_front_of', 'watch', 'next_to', 'watch', 'in_front_of', 'behind', 'lean_on', 
    'above', 'hug', 'speak_to', 'hold_hand_of', 'pull', 'next_to', 'next_to', 'next_to', 'next_to', 'press', 'next_to', 
    'towards', 'towards', 'away', 'next_to', 'hold', 'use', 'watch', 'next_to', 'hold', 'in_front_of', 'lean_on', 'inside', 
    'push', 'pull', 'next_to', 'above', 'behind', 'play_instrument', 'watch', 'ride', 'next_to', 'hold', 'towards', 
    'away', 'away', 'lean_on', 'behind', 'watch', 'use', 'lean_on', 'above', 'next_to', 'away', 'towards', 'in_front_of', 
    'in_front_of', 'above', 'carry', 'above', 'watch', 'in_front_of', 'in_front_of', 'watch', 'behind', 'caress', 'watch', 
    'grab', 'lift', 'touch', 'inside', 'above', 'watch', 'behind', 'away', 'in_front_of', 'next_to', 'towards', 'away', 
    'towards', 'chase', 'behind', 'next_to', 'towards', 'next_to', 'hold', 'caress', 'lean_on', 'above', 'pat', 
    'next_to', 'towards', 'away', 'next_to', 'push', 'touch', 'hold', 'next_to', 'point_to', 'in_front_of', 'next_to', 
    'carry', 'watch', 'next_to', 'carry', 'wave', 'in_front_of', 'wave', 'hug', 'away', 'beneath', 'lean_on', 'lean_on', 
    'pat', 'next_to', 'behind', 'push', 'behind', 'pull', 'watch', 'next_to', 'towards', 'watch', 'hold', 'release', 
    'push', 'next_to', 'watch', 'next_to', 'in_front_of', 'next_to', 'hug', 'kiss', 'feed', 'behind', 'towards', 
    'next_to', 'lift', 'away', 'throw', 'in_front_of', 'behind', 'in_front_of', 'watch', 'watch', 'touch', 'release', 
    'kiss', 'lean_on', 'inside', 'drive', 'chase', 'in_front_of', 'watch', 'away', 'push', 'in_front_of', 'next_to', 
    'shake_hand_with', 'hug', 'in_front_of', 'above', 'ride', 'behind', 'towards', 'away', 'in_front_of', 'next_to', 
    'next_to', 'carry', 'in_front_of', 'in_front_of', 'in_front_of', 'touch', 'next_to', 'behind', 'next_to', 
    'next_to', 'next_to', 'hold', 'clean', 'close', 'grab', 'release', 'beneath', 'hold', 'hold', 'above', 'towards', 
    'carry', 'watch', 'away', 'above', 'ride', 'away', 'kick', 'kick', 'above', 'ride', 'behind', 'in_front_of', 
    'in_front_of', 'towards', 'away', 'pull', 'lean_on', 'pat', 'press', 'wave', 'away', 'towards', 'above', 'ride', 
    'in_front_of', 'feed', 'press', 'squeeze', 'press', 'hit', 'grab', 'hold', 'next_to', 'inside', 'hold', 'grab', 
    'in_front_of', 'pull', 'watch', 'next_to', 'towards', 'feed', 'caress', 'squeeze', 'watch', 'beneath', 'bite', 
    'pull', 'wave', 'in_front_of', 'carry', 'in_front_of', 'behind', 'bite', 'watch', 'next_to', 'open', 'close', 
    'watch', 'next_to', 'press', 'lift', 'shake_hand_with', 'caress', 'towards', 'towards', 'away', 'hold', 
    'wave_hand_to', 'inside', 'drive', 'ride', 'hit', 'hit', 'away', 'towards', 'watch', 'towards', 'point_to', 
    'away', 'touch', 'release', 'touch', 'next_to', 'grab', 'hold', 'watch', 'release', 'touch', 'watch', 'touch', 
    'towards', 'in_front_of', 'away', 'towards', 'away', 'lift', 'grab', 'hold', 'next_to', 'bite', 'watch', 'away', 
    'get_on', 'ride', 'inside', 'beneath', 'away', 'hold', 'behind', 'watch', 'touch', 'feed', 'above', 'ride', 
    'towards', 'push', 'above', 'drive', 'above', 'grab', 'knock', 'next_to', 'hold', 'clean', 'inside', 'watch', 
    'behind', 'release', 'grab', 'lean_on', 'watch', 'next_to', 'hug', 'in_front_of', 'in_front_of', 'behind', 
    'in_front_of', 'towards', 'away', 'hug', 'towards', 'away', 'release', 'grab', 'above', 'ride', 'next_to',
     'watch', 'touch', 'push', 'hold', 'release', 'behind', 'next_to', 'towards', 'shout_at', 'towards', 'towards', 
     'away', 'cut', 'watch', 'behind', 'bite', 'get_off', 'away', 'push', 'next_to', 'watch', 'in_front_of', 'next_to', 
     'away', 'release', 'pull', 'press', 'beneath', 'point_to', 'release', 'behind', 'watch', 'towards', 'get_on', 
     'above', 'ride', 'in_front_of', 'hold', 'lean_on', 'inside', 'next_to', 'ride', 'watch', 'next_to', 'hold', 
     'behind', 'next_to', 'next_to', 'watch', 'hold', 'next_to', 'cut', 'carry', 'next_to', 'drive', 'ride',
      'next_to', 'hold', 'watch', 'wave', 'hit', 'away', 'towards', 'inside', 'behind', 'hold', 'lick', 'caress', 
      'behind', 'towards', 'touch', 'away', 'in_front_of', 'next_to', 'behind', 'towards', 'away', 'ride', 'watch', 
      'release', 'behind', 'towards', 'away', 'towards', 'in_front_of', 'away', 'watch', 'carry', 'behind', 'behind', 
      'use', 'behind', 'hold', 'pull', 'ride', 'away', 'release', 'above', 'release', 'bite', 'grab', 'touch', 'pat', 
      'in_front_of', 'play_instrument', 'lean_on', 'above', 'towards', 'watch', 'away', 'touch', 'away', 'pat', 'get_on', 
      'next_to', 'behind', 'away', 'away', 'in_front_of', 'towards', 'towards', 'away', 'get_off', 'lean_on', 'watch', 
      'away', 'in_front_of', 'watch', 'watch', 'watch', 'behind', 'next_to', 'inside', 'lean_on', 'above', 'touch', 
      'watch', 'above', 'in_front_of', 'press', 'next_to', 'towards', 'pat', 'grab', 'release', 'watch', 'next_to', 
      'pull', 'lift', 'clean', 'pat', 'bite', 'grab', 'open', 'in_front_of', 'towards', 'beneath', 'away', 'chase', 
      'towards', 'caress', 'above', 'wave_hand_to', 'away', 'point_to', 'clean', 'throw', 'watch', 'behind', 'watch', 
      'press', 'inside', 'grab', 'hold', 'release', 'touch', 'wave_hand_to', 'hit', 'speak_to', 'press', 'lift', 'ride', 
      'above', 'away', 'pat', 'watch', 'lift', 'carry', 'release', 'press', 'watch', 'lick', 'beneath', 'pull', 'hug', 
      'press', 'push', 'in_front_of', 'watch', 'behind', 'press', 'lift', 'watch', 'in_front_of', 'above', 'ride', 
      'wave_hand_to', 'behind', 'towards', 'chase', 'next_to', 'away', 'beneath', 'beneath', 'hold', 'away', 'away',
       'towards', 'release', 'in_front_of'
]
# print(len(HOI_actions)) # 557

# VidHOI Dataset labels
verb_dict = {0: 'lean_on', 1: 'watch', 2: 'above', 3: 'next_to', 4: 'behind', 5: 'away', 6: 'towards', 
              7: 'in_front_of', 8: 'hit', 9: 'hold', 10: 'wave', 11: 'pat', 12: 'carry', 13: 'point_to', 
              14: 'touch', 15: 'play_instrument', 16: 'release', 17: 'ride', 18: 'grab', 19: 'lift', 
              20: 'use', 21: 'press', 22: 'inside', 23: 'caress', 24: 'pull', 25: 'get_on', 26: 'cut', 
              27: 'hug', 28: 'bite', 29: 'open', 30: 'close', 31: 'throw', 32: 'kick', 33: 'drive', 
              34: 'get_off', 35: 'push', 36: 'wave_hand_to', 37: 'feed', 38: 'chase', 39: 'kiss', 
              40: 'speak_to', 41: 'beneath', 42: 'smell', 43: 'clean', 44: 'lick', 45: 'squeeze',
              46: 'shake_hand_with', 47: 'knock', 48: 'hold_hand_of', 49: 'shout_at'}

obj_dict = {0: 'person', 1: 'car', 2: 'guitar', 3: 'chair', 4: 'handbag', 5: 'toy', 6: 'baby_seat', 
            7: 'cat', 8: 'bottle', 9: 'backpack', 10: 'motorcycle', 11: 'ball/sports_ball', 
            12: 'laptop', 13: 'table', 14: 'surfboard', 15: 'camera', 16: 'sofa', 
            17: 'screen/monitor', 18: 'bicycle', 19: 'vegetables', 20: 'dog', 21: 'fruits', 
            22: 'cake', 23: 'cellphone', 24: 'cup', 25: 'bench', 26: 'snowboard', 27: 'skateboard', 
            28: 'bread', 29: 'bus/truck', 30: 'ski', 31: 'suitcase', 32: 'stool', 33: 'bat', 
            34: 'elephant', 35: 'fish', 36: 'baby_walker', 37: 'dish', 38: 'watercraft', 
            39: 'scooter', 40: 'pig', 41: 'refrigerator', 42: 'horse', 43: 'crab', 44: 'bird', 
            45: 'piano', 46: 'cattle/cow', 47: 'lion', 48: 'chicken', 49: 'camel', 
            50: 'electric_fan', 51: 'toilet', 52: 'sheep/goat', 53: 'rabbit', 54: 'train', 
            55: 'penguin', 56: 'hamster/rat', 57: 'snake', 58: 'frisbee', 59: 'aircraft', 
            60: 'oven', 61: 'racket', 62: 'faucet', 63: 'antelope', 64: 'duck', 65: 'stop_sign', 
            66: 'sink', 67: 'kangaroo', 68: 'stingray', 69: 'turtle', 70: 'tiger', 
            71: 'crocodile', 72: 'bear', 73: 'microwave', 74: 'traffic_light', 75: 'panda',
            76: 'leopard', 77: 'squirrel'}

json_file = '/home/da/data/Dataset/Vid_HOI/train_frame_annots.json'
with open(json_file, 'r') as f:
    label = json.load(f)


from torchvision.ops.boxes import _box_inter_union

def giou_loss(input_boxes, target_boxes, eps=1e-7):
    """
    Args:
        input_boxes: Tensor of shape (N, 4) or (4,).
        target_boxes: Tensor of shape (N, 4) or (4,).
        eps (float): small number to prevent division by zero
    """
    inter, union = _box_inter_union(input_boxes, target_boxes)
    iou = inter / union

    # area of the smallest enclosing box
    min_box = torch.min(input_boxes, target_boxes)
    max_box = torch.max(input_boxes, target_boxes)
    area_c = (max_box[:, 2] - min_box[:, 0]) * (max_box[:, 3] - min_box[:, 1])
    giou = iou - ((area_c - union) / (area_c + eps))
    loss = 1 - giou

    return loss.sum()


class HOIClssificationBboxReg(nn.Module):
    def __init__(
        self, 
        node_dim: int = 16
    ):
        super().__init__()
        self.node_dim = node_dim
        # self.train_frame_annots = train_frame_annots
        self.mlp_pred = mlp.MLP(2 * self.node_dim, self.node_dim // 4, 1, 3) 
        self.threshold = nn.Parameter(0.1 * torch.ones(len(actions)))
        self.mu1 = 2.5
        self.mu2 = 1
        
    def forward(self, action_prompts: torch.Tensor, HOI_prompts: torch.Tensor, frame_name: str, 
                comb_features: torch.Tensor, comb_bboxes: torch.Tensor):
        num_act = len(actions)
        num_hoi = len(HOI_actions)
        num_comb_fea = comb_features.shape[0]

        # predictions of HOI logits for each frame
        HOI_logits = []
        for i in range(num_act):
            act_name = actions[i]
            for j in range(num_hoi):
                hoi_act_name = HOI_actions[j]
                if act_name == hoi_act_name:
                    # print(action_prompts[i].shape) # torch.Size([16])
                    # print(HOI_prompts[i].shape) # torch.Size([16])
                    _HOI_prompt = torch.cat((action_prompts[i].unsqueeze(0), HOI_prompts[j].unsqueeze(0)), 1)
                    # print(_HOI_prompt.shape) # torch.Size([1, 32])
                    # logit = F.sigmoid(self.mlp_pred(_HOI_prompt)).detach().numpy()[0]
                    logit = torch.sigmoid(self.mlp_pred(_HOI_prompt)).item()
                    # print("logit: {}".format(logit))
                    HOI_logits.append(logit)
        # print("HOI logits: ", HOI_logits)

        HOI_pred_idx = np.where(np.asarray(HOI_logits) > 0.5)[0] # HOI classifcation
        pred_act_list, pred_act_idx_list = [], []
        for i in range(len(HOI_pred_idx)):
            pred_HOI_name = HOIs[HOI_pred_idx[i]]
            pred_act_name = pred_HOI_name.split(",")[1]
            # print("pred action name: ", pred_act_name)
            if pred_act_name not in pred_act_list:
                pred_act_list.append(pred_act_name)
                idx = actions.index(pred_act_name)
                pred_act_idx_list.append(idx)
        print("pred_act_list: ", pred_act_list)
        print("pred_act_idx_list: ", pred_act_idx_list)

        # ground truth of each frame
        gt = torch.zeros(num_hoi)
        gt_act_list = [] # action idnexes in ground truth
        # gt_box = [] # ground truth bboxes of both human and object
        gt_act_name_list = []
        gt_box = [] # ground truth bboxes of both human and object
        gt_box_dict = {}
        pred_bbox_dict = {}
        gt_bbox_dict = {"0": [[[5,7,35,76], [3,9,24,56]]]}
        pred_bbox_dict = {"0": [[[4, 3, 65, 45], [12, 3, 45, 67]], [[14, 13, 165, 145], [15, 23, 65, 167]]]}
        for i in range(len(label)):
            gt_hu_bbox, gt_ob_bbox, box = [], [], []
            gt_name = join(
                label[i]['video_folder'], label[i]['video_id'], label[i]['video_id'] + '_' +
                 label[i]['frame_id'] + '.jpg'
            )
            # print(name)
            if frame_name == gt_name:
                HOI_label = 'person,' + verb_dict[label[i]['action_class']] + ',' + obj_dict[label[i]['object_class']]
                print("HOI label: ", HOI_label)
                # gt_box = [] # ground truth bboxes of both human and object
                if HOI_label in HOIs:
                    idx = HOIs.index(HOI_label) 
                    # print(idx)
                    gt[idx] = 1
                    gt_act_name = HOI_actions[idx] # the ground-truth action name
                    idx = actions.index(gt_act_name) # ground-truth action index in actions list
                    if idx not in gt_act_list:
                        gt_act_list.append(idx)
                        gt_act_name_list.append(gt_act_name)
            
                for n in gt_act_name_list:
                    # gt_hu_bbox, gt_ob_bbox, box = [], [], []
                    gt_box_dict[f'{n}'] = []
                    if verb_dict[label[i]['action_class']] == n:
                        gt_hu_bbox.append(label[i]['person_box']['xmin'])
                        gt_hu_bbox.append(label[i]['person_box']['ymin'])
                        gt_hu_bbox.append(label[i]['person_box']['xmax'])
                        gt_hu_bbox.append(label[i]['person_box']['ymax'])
                        box.append(gt_hu_bbox)
                        gt_ob_bbox.append(label[i]['object_box']['xmin'])
                        gt_ob_bbox.append(label[i]['object_box']['ymin'])
                        gt_ob_bbox.append(label[i]['object_box']['xmax'])
                        gt_ob_bbox.append(label[i]['object_box']['ymax'])
                        box.append(gt_ob_bbox)
                        gt_box.append(box)
                    # print("length of ground truth box: ", len(gt_box))
                    gt_box_dict[f'{n}'].append(gt_box)
        # gt_box_dict[f'{n}'] = gt_box

        # print("Ground Truth: ", gt)
        print('-' * 50)
        print("gt_act_list: ", gt_act_list)
        print('-' * 50)
        print("gt_box: ", gt_box)
        print('-' * 50)
        print("gt_box_dict: ", gt_box_dict)
        print('-' * 50)
        print("gt_act_name_list: ", gt_act_name_list)

        cos = CosineSimilarity(dim = 1, eps=1e-6)
        pred_comb_dict = {}

        for i in gt_act_list:
            pred_comb = []
            for j in range(num_comb_fea):
                sim = torch.abs(cos(comb_features[j].unsqueeze(0), action_prompts[i].unsqueeze(0)))
                print("similarities between action prompts and comb features: ", sim)
                if sim > self.threshold[i]:
                    pred_comb.append(j)
            pred_comb_dict['{}'.format(actions[i])] = pred_comb
        print('-' * 50)
        print("pred_comb_dict: ", pred_comb_dict)
        print("thresholds: ", self.threshold)

        r"""HOI classification loss"""
        bce = nn.BCEWithLogitsLoss()
        Loss_hoi_cls = bce(torch.as_tensor(HOI_logits), gt)
        Loss_bbox_cls, Loss_bbox_loc = 0, 0

        assert len(pred_bbox_dict) == len(gt_bbox_dict)
        print(" num of pred_bbox_dict: ", len(pred_bbox_dict))
        for i in range(len(gt_bbox_dict)):
            gt_bboxes = gt_bbox_dict["{}".format(i)]
            pred_bboxes = pred_bbox_dict["{}".format(i)]
            print("pred bboxes: ", torch.tensor(pred_bboxes)) 
            print("pred bboxes: ", torch.tensor(pred_bboxes[1][0]).shape)
            print("pred bboxes: ", torch.tensor(pred_bboxes[1][0]))
            pred_person_bboxes = torch.cat(
                (torch.tensor(pred_bboxes[0][0]).unsqueeze(0), torch.tensor(pred_bboxes[1][0]).unsqueeze(0)), 
                dim = 0
            )
            print("pred person bboxes: ", pred_person_bboxes)
            print("pred person bboxes shape: ", pred_person_bboxes.shape) # torch.Size([2, 4])
            gt_person_bboxes = torch.tensor(gt_bboxes[0][0]).unsqueeze(0)
            print("gt person bboxes: ", gt_person_bboxes)
            print("gt person bboxes shape: ", gt_person_bboxes.shape) # torch.Size([1, 4])
            IoUs = box_iou(pred_person_bboxes, gt_person_bboxes)
            print("IoUs: ", IoUs)
            print("IoUs: ", IoUs.shape)
            if len(gt_bboxes) > len(pred_bboxes):
                length = len(gt_bboxes)
                binary_gt = torch.ones(length)
                binary_pred = torch.cat((torch.ones(len(pred_bboxes)), torch.zeros(length - len(pred_bboxes))), dim = 0)
                L_bbox_cls = bce(binary_pred, binary_gt)
                Loss_bbox_cls += L_bbox_cls
            elif len(gt_bboxes) < len(pred_bboxes):
                length = len(pred_bboxes)
                binary_gt = torch.cat((torch.ones(len(pred_bboxes)), torch.zeros(length - len(pred_bboxes))), dim = 0)
                binary_pred = torch.ones(length)
                L_bbox_cls = bce(binary_pred, binary_gt)
                Loss_bbox_cls += L_bbox_cls
            else:
                L_bbox_cls = 0

            r"""
            bbox_overlap: 
                ||\hat{b}_i^{(h)} - \hat{b}_{\omega_i}^{(h)}|| + ||\hat{b}_i^{(o)} - \hat{b}_{\omega_i}^{(o)}||
            """
            sorted_IoU_idx = torch.argsort(IoUs, dim=0, descending = True)
            print("sorted IoU indexes: ", sorted_IoU_idx)
            # bbox_overlap = torch.norm()
            # giou = giou_loss() + giou_loss()
            # Loss_bbox_loc = self.mu1 * bbox_overlap + self.mu2 * giou

        Loss_bbox_cls /= len(gt_bbox_dict)
        Loss_bbox_loc /= len(gt_bbox_dict)

        return Loss_hoi_cls, Loss_bbox_cls, Loss_bbox_loc

