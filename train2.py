import os
import time
import torch
import torch.cuda as cuda
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader
import sys
# sys.path.append(gdrive_dir)
from dataset import VOCDataset, collate_fn
from ssdEff import SSDEff
from loss import MultiBoxLoss
from utils import create_data_lists, clip_gradient, save_checkpoint
from eval_detection_voc import eval_detection_voc
device = torch.device("cuda" if cuda.is_available() else "cpu") 
print(device)
gdrive_dir = '/media/hanoi1/mount_disk/Projects/single-shot-detection-pytorch'
print(gdrive_dir)

import torch.nn as nn
import torch.nn.functional as F
from utils import decimate, xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, find_jaccard_overlap

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, x, y):
        ''' 
        x: (N, C)
        y: (N,)
        '''
        n_classes = x.size(1)
        
        # Convert y to one hot embedding 
        t = torch.eye(n_classes).to(device)[y] # (N,21)

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)
        w = self.alpha*t + (1-self.alpha)*(1-t)
        w = w * (1-pt).pow(self.gamma)
        
        return F.binary_cross_entropy_with_logits(x, t, w.detach_(), reduction=self.reduction)


class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1., focal_loss=False):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        
        # loss functions
        self.smooth_l1 = nn.SmoothL1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.focal_loss = FocalLoss(reduction='sum') if focal_loss else None
        
    def match_gt_priors(self, boxes, labels):
        ''' Given gt boxes, labels and (8732) priors, match them into the most suited priors
        N: batch size
        Params:
            boxes: true object bounding boxes in boundary coordinates, (xy), a list of N tensors: N(n_objects, 4)
            labels: true object labels, a list of N tensors: N(n_objects,)
        Return: 
            truth_offsets: tensor (N, 8732, 4)
            truth_classes: tensor (N, 8732,)
        '''
        N = len(boxes) #batch size
        n_priors = self.priors_cxcy.size(0)
        # print(n_priors)
        
        truth_offsets = torch.zeros((N, n_priors, 4), dtype=torch.float).to(device)
        truth_classes = torch.zeros((N, n_priors), dtype=torch.long).to(device)
        
        # for each image
        for i in range(N):
            n_objects = labels[i].shape[0]

            overlap = find_jaccard_overlap(self.priors_xy, boxes[i]) #(n_priors, n_boxes)
            # print(overlap, overlap.shape)
            
            # for each prior, find the max iou and the coresponding object id
            prior_iou, prior_obj = overlap.max(dim=1) #(n_priors)
            # print(prior_iou, prior_obj)
            
            # for each object, find the most suited prior id
            _, object_prior = overlap.max(dim=0) #(n_objects)
            # print(_, object_prior)
            # for each object, assign its most suited prior with object id 
            for j in range(n_objects): prior_obj[object_prior[j]] = j
            # for each object, assign its most suited prior with hight iou to ensure it qualifies the thresholding 
            prior_iou[object_prior] = 1.
            
            # match bbox coordinates
            boxes_xy = boxes[i][prior_obj] # (8732, 4)
            # print(boxes[0].shape, prior_obj, boxes_xy.shape)
            
            # match prior class
            prior_class = labels[i][prior_obj]  # (8732)
            # thresholding: assign prior with iou < threshold to the class 0: background
            prior_class[prior_iou < self.threshold] = 0
            
            # save into the truth tensors
            truth_offsets[i,:,:] = cxcy_to_gcxgcy(xy_to_cxcy(boxes_xy), self.priors_cxcy)
            truth_classes[i,:] = prior_class
        
        return truth_offsets, truth_classes
    
    def forward(self, predicted_offsets, predicted_scores, boxes, labels):
        '''
        Params:
            predicted_offsets: predicted offsets w.r.t the 8732 prior boxes, (gcxgcy), a tensor of dimensions (N, 8732, 4)
            predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
            boxes: true  object bounding boxes in boundary coordinates, (xy), a list of N tensors: N(n_objects, 4)
            labels: true object labels, a list of N tensors: N(n_objects,)
        Return: 
            multibox loss, a scalar
        '''
        N = predicted_offsets.shape[0]
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        
        # Matching ground truth boxes (N, n_objects,4) to priors (N, 8732, 4)
        truth_offsets, truth_classes = self.match_gt_priors(boxes, labels)
        
        # Now we have ground truth priors and predicted priors we can now calculate the loss
        positive_priors = (truth_classes != 0) #(N,8732)
        n_positives = positive_priors.sum()  # (1)
        
        # Calculating loss = alpha*loc_loss + conf_loss
        # loc_loss: localization loss
        loc_loss = self.smooth_l1(predicted_offsets[positive_priors], truth_offsets[positive_priors])
        
        # Confidence loss
        if self.focal_loss is not None:
            conf_loss = self.focal_loss(predicted_scores.view(-1, n_classes), truth_classes.view(-1)) / n_positives
        else:# Hard negative mining
            full_conf_loss = self.cross_entropy(predicted_scores.view(-1, n_classes), truth_classes.view(-1)) #(N*n_priors)
            full_conf_loss = full_conf_loss.view(N, n_priors)
            # Since there is a huge unbalance between positive and negative priors so we only take the loss of the hard negatives

            n_hard_negatives = self.neg_pos_ratio * positive_priors.sum(dim=1)  # (N)
            conf_loss_hard_neg = 0
            # accummulate conf_loss_hard_neg for each sample in batch
            for i in range(N):
                conf_loss_neg,_ = full_conf_loss[i][~positive_priors[i]].sort(dim=0, descending=True)
                conf_loss_hard_neg = conf_loss_hard_neg + conf_loss_neg[0:n_hard_negatives[i]].sum()

            conf_loss = (full_conf_loss[positive_priors].sum() + conf_loss_hard_neg) / n_positives
        
        #print(loc_loss.item(), conf_loss.item())
        return self.alpha * loc_loss + conf_loss

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from aug import SSDAugmentation, SSDTransform
from utils import rescale_coordinates, save_aug


class VOCDataset(Dataset):
    def __init__(self, data_folder, json_files, augment=False, keep_difficult=False, img_size=300):
        super(VOCDataset, self).__init__()
        self.root = data_folder
        self.keep_difficult = keep_difficult
        self.img_size = img_size
        self.transform = SSDTransform(size=img_size)
        self.augment = SSDAugmentation(size=img_size) if augment else None
        
        self.img_paths = list()
        self.targets = list()
        
        with open(os.path.join(self.root, json_files[0]), 'r') as f:
            self.img_paths = json.load(f)
        with open(os.path.join(self.root, json_files[1]), 'r') as f:
            self.targets = json.load(f)
        assert len(self.img_paths) == len(self.targets)
        
        if not self.keep_difficult:
            self.remove_difficult_objs()
    
    def __len__(self):
        return len(self.img_paths)
    
    def remove_difficult_objs(self):
        for target in self.targets:
            boxes = target['boxes']
            labels = target['labels']
            difficulties = target['difficulties']
            
            # remove difficult objects if not keep_difficult
            boxes =  [boxes[i] for i in range(len(boxes)) if not difficulties[i]]
            labels = [labels[i] for i in range(len(labels)) if not difficulties[i]]
    
    def __getitem__(self, index):
        img = self.read_img(index)
        target = self.targets[index]
        boxes = target['boxes'].copy()
        labels = target['labels'].copy()
        
        if self.augment is not None:
            img, boxes, labels = self.augment(img, boxes, labels)
            if len(boxes) < 1: 
                # This is the case when albumentations crop or shrink all the objects and discard all of them
                # So in this case(~0.5%), no augmentation will be done.
                img, boxes, labels = self.read_img(index), target['boxes'].copy(), target['labels'].copy()
            
        img, boxes, labels = self.transform(img, boxes, labels)
            
        #save_aug(img, boxes, labels, os.path.basename(self.img_paths[index]))
        # to tensor
        img = torch.Tensor(img.transpose((2,0,1)))
        boxes = rescale_coordinates(boxes, h=img.size(1), w=img.size(2))
        labels = torch.IntTensor(labels)
        
        return img, boxes, labels
    
    def read_img(self, index):
        ''' read the image and convert to RGB '''
        # print(self.img_paths[index])
        # print(cv2.imread(self.img_paths[index]).shape)
        # return cv2.cvtColor(cv2.imread(self.img_paths[index]), cv2.COLOR_BGR2RGB)
        idx = index
        while 1:
             try:
                return cv2.cvtColor(cv2.imread(self.img_paths[index]), cv2.COLOR_BGR2RGB)
             except:
                idx = (idx+1)/len(self.img_paths)


def collate_fn(batch):
    """ Explaination
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    Note: this need not be defined in this Class, can be standalone.

    Param: batch: an iterable of N sets from __getitem__()
    Return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """
    batch_imgs = list()
    batch_boxes = list()
    batch_labels = list()

    for imgs, boxes, labels in batch:
        batch_imgs.append(imgs)
        batch_boxes.append(boxes)
        batch_labels.append(labels)

    batch_imgs = torch.stack(batch_imgs, dim=0)

    return batch_imgs, batch_boxes, batch_labels

batch = 1
torch.manual_seed(42)
trainset = VOCDataset(data_folder=gdrive_dir+'/data/', json_files=('TRAIN_images.json', 'TRAIN_objects.json'), augment=True, img_size=300)
valset = VOCDataset(data_folder=gdrive_dir+'/data/', json_files=('VAL_images.json', 'VAL_objects.json'), img_size=300)

dataloaders = dict(
    train = DataLoader(trainset, batch_size=batch, collate_fn=collate_fn, shuffle=True, num_workers=0),
    val = DataLoader(valset, batch_size=batch, collate_fn=collate_fn, shuffle=False, num_workers=0),
)


import torch
import torch.nn as nn
import torch.nn.functional as F
from effnet.efficient_net_b3 import EfficientNet
from torchvision.ops import nms
from utils import cxcy_to_xy, gcxgcy_to_cxcy, create_prior_boxes


EXTRAS = {
    'efficientnet-b3': [
        # in,  out, k, s, p
        [(384, 128, 1, 1, 0), (128, 256, 3, 2, 1)],  # 5 x 5
        [(256, 128, 1, 1, 0), (128, 256, 3, 1, 0)],  # 3 x 3
    ]
}

def add_extras(cfgs):
    extras = nn.ModuleList()
    for cfg in cfgs:
        extra = []
        for params in cfg:
            in_channels, out_channels, kernel_size, stride, padding = params
            extra.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            extra.append(nn.ReLU())
        extras.append(nn.Sequential(*extra))
    return extras


class SSDEff(nn.Module):
    def __init__(self, n_classes):
        super(SSDEff, self).__init__()
        self.n_classes = n_classes
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        ''' Efficient-net-b3 outputs
        fm38: N, 48, 38, 38
        fm19: N, 136, 19, 19
        fm10: N, 384, 10, 10
        '''
        self.extras = add_extras(EXTRAS['efficientnet-b3'])
        ''' extras conv outputs
        fm5: N, 256, 5, 5
        fm3: N, 256, 3, 3
        '''
        
        '''
        # FPN Lateral layers
        self.lat_fm19 = nn.Conv2d(136, 256, kernel_size=3, padding=1)
        self.lat_fm38 = nn.Conv2d(48, 256, kernel_size=3, padding=1)
        
        # FPN Top-down layers
        self.final_fm10 = nn.Conv2d(384, 256, kernel_size=1, padding=0)
        self.final_fm19 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.final_fm38 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        '''
        
        # FPN layers
        self.lat_fm10 = nn.Conv2d(384, 136, kernel_size=1, padding=0)
        self.lat_fm19 = nn.Conv2d(136, 48, kernel_size=1, padding=0)
        self.final_fm19 = nn.Conv2d(136, 136, kernel_size=3, stride=1, padding=1)
        self.final_fm38 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        
        # Detection layers
        self.det_fm38 = nn.Conv2d(48, 4*(4+n_classes), kernel_size=3, padding=1)
        self.det_fm19 = nn.Conv2d(136, 6*(4+n_classes), kernel_size=3, padding=1)
        self.det_fm10 = nn.Conv2d(384, 6*(4+n_classes), kernel_size=3, padding=1)
        self.det_fm5 = nn.Conv2d(256, 6*(4+n_classes), kernel_size=3, padding=1)
        self.det_fm3 = nn.Conv2d(256, 4*(4+n_classes), kernel_size=3, padding=1)
        
        self.init_weights()
        self.priors_cxcy = self.get_prior_boxes()

    def init_weights(self):
        #Init weights for detection layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        ''' x: tensor (n, 3, 300, 300)
        '''
        n = x.size(0)
        
        # Bottom-up
        fm38, fm19, fm10 = self.backbone.extract_feature_maps(x)
        fm5 = self.extras[0](fm10)
        fm3 = self.extras[1](fm5)
        
        '''
        # Top-down + lateral connections
        fm10 = F.relu(self.final_fm10(fm10))
        
        fm19 = F.relu(self.lat_fm19(fm19)) + F.interpolate(fm10, size=(19,19), mode='nearest')
        fm19 = F.relu(self.final_fm19(fm19))
        
        fm38 = F.relu(self.lat_fm38(fm38)) + F.interpolate(fm19, size=(38,38), mode='nearest')
        fm38 = F.relu(self.final_fm38(fm38))
        '''
        fm19 = fm19 + F.interpolate(F.relu(self.lat_fm10(fm10)), size=(19,19), mode='nearest')
        fm19 = F.relu(self.final_fm19(fm19))
        
        fm38 = fm38 + F.interpolate(F.relu(self.lat_fm19(fm19)), size=(38,38), mode='nearest')
        fm38 = F.relu(self.final_fm38(fm38))
        

        # Detection
        box_size = 4 + self.n_classes  # each box has 25 values: 4 offset values and 21 class scores
        #
        det_fm38 = self.det_fm38(fm38)
        det_fm38 = det_fm38.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 5776, box_size)
        
        det_fm19 = self.det_fm19(fm19)
        det_fm19 = det_fm19.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 2166, box_size)
        
        det_fm10 = self.det_fm10(fm10)
        det_fm10 = det_fm10.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 600, box_size)
        
        det_fm5 = self.det_fm5(fm5)
        det_fm5 = det_fm5.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 150, box_size)
        
        det_fm3 = self.det_fm3(fm3)
        det_fm3 = det_fm3.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 36, box_size)
        
        detection = torch.cat([det_fm38, det_fm19, det_fm10, det_fm5, det_fm3], dim=1)  # (N, 8692, box_size)
        offsets, class_scores = torch.split(detection, [4,self.n_classes], dim=2)
        
        return offsets, class_scores
    
    
    def get_prior_boxes(self):
        '''
        Return: 
            prior boxes in center-size coordinates, a tensor of dimensions (n_priors, 4)
        '''
        fmap_dims = {'fm38': 38,
                     'fm19': 19,
                     'fm10': 10,
                     'fm5': 5,
                     'fm3': 3}

        obj_scales = {'fm38': 0.08,
                      'fm19': 0.16,
                      'fm10': 0.32,
                      'fm5': 0.54,
                      'fm3': 0.75}

        aspect_ratios = {'fm38': [1., 2., 0.5],
                         'fm19': [1., 2., 3., 0.5, .333],
                         'fm10': [1., 2., 3., 0.5, .333],
                         'fm5': [1., 2., 3., 0.5, .333],
                         'fm3': [1., 2., 0.5]}

        return create_prior_boxes(fmap_dims, obj_scales, aspect_ratios, last_scale=0.85)
    
    
    def post_process_top_k(self, predicted_offsets, predicted_scores, score_threshold, iou_threshold, top_k):
        ''' return top_k detections sorted by confidence score
        Params:
            predicted_offsets: predicted offsets w.r.t the prior boxes, (gcxgcy), a tensor of dimensions (N, 8692, 4)
            predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8692, n_classes)
            score_threshold: minimum threshold for a box to be considered a match for a certain class
            iou_threshold: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
            top_k: int, if the result contains more than k objects, just return k objects that have largest confidence score
        Return:
            detections: (boxes, labels, and scores), they are lists of N tensors
            boxes: N (n_boxes, 4)
            labels: N (n_boxes,)
            scores: N (n_boxes,)
        '''
        boxes = list()
        labels = list()
        scores = list()
        N, n_priors = predicted_offsets.shape[0:2]
        
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8692, n_classes)
        
        # for each image in the batch
        for i in range(N):
            boxes_i = list()
            labels_i = list()
            scores_i = list()
            
            # convert gcxgcy to xy coordinates format
            boxes_xy = cxcy_to_xy(gcxgcy_to_cxcy(predicted_offsets[i], self.priors_cxcy)) # (8692, 4)

            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8692)
                qualify_mask = class_scores > score_threshold
                n_qualified = qualify_mask.sum().item()
                if n_qualified == 0:
                    continue
                boxes_class_c = boxes_xy[qualify_mask]  # (n_qualified, 4)
                boxes_score_class_c = class_scores[qualify_mask]  # (n_qualified) <= 8692
                
                final_box_ids = nms(boxes_class_c, boxes_score_class_c, iou_threshold)  # (n_final_boxes,)
                
                boxes_i.extend(boxes_class_c[final_box_ids].tolist())
                labels_i.extend([c]*len(final_box_ids))
                scores_i.extend(boxes_score_class_c[final_box_ids].tolist())
        
            boxes.append(torch.FloatTensor(boxes_i).to(device))
            labels.append(torch.LongTensor(labels_i).to(device))
            scores.append(torch.FloatTensor(scores_i).to(device))
            
            # Filter top k objects that have largest confidence score
            if boxes[i].size(0) > top_k:
                scores[i], sort_ind = scores[i].sort(dim=0, descending=True)
                scores[i] = scores[i][:top_k]  # (top_k)
                boxes[i] = boxes[i][sort_ind[:top_k]]  # (top_k, 4)
                labels[i] = labels[i][sort_ind[:top_k]]  # (top_k)

        return boxes, labels, scores

checkpoint_path = gdrive_dir+'/pretrained/mine.pt'
# checkpoint = torch.load(checkpoint_path) # None

if False:
    ssd_eff = checkpoint['model']
    optimizer = checkpoint['optimizer']
    #exp_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=checkpoint['epoch']-1)
else:
    print('New model')
    ssd_eff = SSDEff(n_classes=15).to(device)
    optimizer = torch.optim.Adam(ssd_eff.parameters(), lr=2e-3, weight_decay=5e-4)
    #exp_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

loss_func = MultiBoxLoss(priors_cxcy=ssd_eff.get_prior_boxes(), threshold=0.5, alpha=1., neg_pos_ratio=3, focal_loss=False)

grad_clip = None

def train_epoch(model, trainset_loader, loss_func, optimizer, epoch_id):
    model.train()
    train_loss = 0
    for step, (imgs, boxes, labels) in enumerate(trainset_loader):

        # print(type(imgs),imgs.shape, imgs)
        # print(type(boxes),boxes[0].shape, boxes)
        # print(type(labels),labels[0].shape, labels)
        # break
        # move input data to GPU
        imgs = imgs.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        # print(type(boxes),boxes[0].shape, boxes)
        # print(type(labels),labels[0].shape, labels)
        # break
        # forward
        predicted_offsets, predicted_scores = model(imgs)
        # print(predicted_offsets.shape, predicted_scores.shape)
        # print('*'*10)
        loss = loss_func(predicted_offsets, predicted_scores, boxes, labels)
        # break
        # backward & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch_id == 1 and step % 50 == 0:
            print(f'Epoch 1 - step {step}: train_loss: {loss.item():.4f}')
        
        train_loss += loss.item()
        
    return round(train_loss/len(trainset_loader), 4)

def eval_epoch(model, valset_loader, loss_func):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for step, (imgs, boxes, labels) in enumerate(valset_loader):
            imgs = imgs.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            
            predicted_offsets, predicted_scores = model(imgs)
            loss = loss_func(predicted_offsets, predicted_scores, boxes, labels)
            val_loss += loss.item()

    return round(val_loss/len(valset_loader), 4)


def train_model(model, dataloaders, optimizer, loss_func, lr_scheduler=None, n_epoch=1, checkpoint=None):
    last_epoch = 0
    if checkpoint:
        last_epoch = checkpoint['epoch']

    train_loss_hist, val_loss_hist = [], []
    for epoch_id in range(last_epoch+1, last_epoch+n_epoch+1):
        start_time = time.time()
        print('Epoch:',epoch_id)
        train_loss = train_epoch(model, dataloaders['train'], loss_func, optimizer, epoch_id)
        # return None
        training_time = round(time.time() - start_time)
        if lr_scheduler: lr_scheduler.step()
        checkpoint_path = gdrive_dir+'/pretrained/'+str(epoch_id)+'mine.pt'
        save_checkpoint(epoch_id, model, optimizer, checkpoint_path)
        
        # eval val_loss every epoch
        start_time = time.time()
        val_loss = eval_epoch(model, dataloaders['val'], loss_func) # if epoch_id % 5 == 0 else 'N/A'
        val_time = round(time.time() - start_time)
        
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        
        print(f'Epoch {epoch_id} - train/val_time: {training_time}s | {val_time}s - train_loss: {train_loss} - val_loss: {val_loss}')
        
    return train_loss_hist, val_loss_hist

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

loss_func = MultiBoxLoss(priors_cxcy=ssd_eff.get_prior_boxes(), threshold=0.5, alpha=1., neg_pos_ratio=3, focal_loss=False)
train_model(ssd_eff, dataloaders, optimizer, loss_func, n_epoch=100)