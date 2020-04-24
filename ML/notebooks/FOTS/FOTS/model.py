import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import FOTS.utils as utils

class SharedConv(nn.Module):
    def __init__(self):
        super(SharedConv, self).__init__()
        self.bbnet = torchvision.models.resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(3072, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(640, 64, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(320, 32, 1)
        self.bn3 = nn.BatchNorm2d(32)
        
    def __bbnetforward(self,x):
        for name,layer in self.bbnet.named_children():
            x = layer(x)
            
            if name=='layer1':
                down1 = x
            elif name=='layer2':
                down2 = x
            elif name=='layer3':
                down3 = x
            elif name =='layer4':
                down4 = x
                break
                
        # torch.Size([1, 256, 64, 64])
        # torch.Size([1, 512, 32, 32])
        # torch.Size([1, 1024, 16, 16])
        # torch.Size([1, 2048, 8, 8])   
        return down1, down2, down3, down4
        
    def forward(self, x):
        down1, down2, down3, down4 = self.__bbnetforward(x)
        
        o = F.interpolate(down4,scale_factor=2, mode='bilinear', align_corners=True)
        o = torch.cat([o,down3], dim=1) # torch.Size([1, 3072, 16, 16])
        o = self.conv1(o)
        o = self.bn1(o)
        o = F.relu(o) # torch.Size([1, 128, 16, 16])
        
        o = F.interpolate(o,scale_factor=2, mode='bilinear', align_corners=True)
        o = torch.cat([o,down2], dim=1) # torch.Size([1, 3584, 32, 32])
        o = self.conv2(o)
        o = self.bn2(o)
        o = F.relu(o) # torch.Size([1, 64, 32, 32])
        
        
        o = F.interpolate(o,scale_factor=2, mode='bilinear', align_corners=True)
        o = torch.cat([o,down1], dim=1) # torch.Size([1, 3840, 64, 64])
        o = self.conv3(o)
        o = self.bn3(o)
        o = F.relu(o) # torch.Size([1, 32, 64, 64])
        
        return o

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.score = nn.Conv2d(32, 1, 1)
        self.geo = nn.Conv2d(32, 4, 1)
        self.angle = nn.Conv2d(32, 1, 1)
        
    def forward(self, x):
        score = self.score(x)
        score = torch.sigmoid(score)
        
        geo = self.geo(x)
        geo = torch.sigmoid(geo) * 512
        
        angle = self.angle(x)
        angle = torch.sigmoid(angle)
        angle = (angle - 0.5) * np.pi / 2
        
        geometry = torch.cat([geo, angle], dim=1)
        
        return score, geometry

class RoiRotate(nn.Module):

    def __init__(self):
        super(RoiRotate, self).__init__()

        
    def forward(self, *inputs):
        feature_map, pred_boxes = inputs
        feature_map = feature_map.detach().numpy().transpose(0, 2, 3, 1) # [n,128,128,32]
        pred_boxes /= 4  #  512 -> 128

        print(feature_map.shape, pred_boxes.shape)
        
        batch_roi = []
        for index, boxes in enumerate(pred_boxes):
            feature = feature_map[index]  # [128,128,32]
            rois = []
            maxwidth = 0
            for box in boxes:
                box = box.reshape(4, 2)
                
                rotate_rect = cv2.minAreaRect(box)
                box_w, box_h = rotate_rect[1][0], rotate_rect[1][1]

                if box_h > box_w:
                    box_w, box_h = box_h, box_w
                
                
                box_w = np.ceil(8 / box_h * box_w).astype(np.int32)
                box_h = 8 # crnn input height 32/4=8
                maxwidth = box_w if box_w > maxwidth else maxwidth
                
                spts = np.array([(box[0][0], box[0][1]),(box[1][0], box[1][1]),(box[3][0], box[3][1])])
                dpts = np.array([(0, 0), (box_w, 0), (0, 8)])

                matrix = cv2.getAffineTransform(spts.astype(np.float32), dpts.astype(np.float32))
                feature_affine = cv2.warpAffine(feature, matrix, (feature.shape[0], feature.shape[1]))\

                roi = feature_affine[:box_h,:box_w,:]
                roi = torch.from_numpy(roi)
                print(roi.shape)
                rois.append(roi)


            batch_roi.append(rois)
        
        return batch_roi

class EAST(nn.Module):
    def __init__(self):
        super(EAST, self).__init__()
        self.sharedConv = SharedConv()
        self.detector = Detector()
        self.roiratate = RoiRotate()
        
    def forward(self, inputs):
        image = inputs

        feature_map = self.sharedConv.forward(image) # [b,32,128,128]
        score_map, geo_map = self.detector(feature_map)  # [b,1,128,128] [b,5,128,128]

        score = score_map.detach().cpu().numpy()
        geometry = geo_map.detach().cpu().numpy()

        pred_boxes = []
        for i in range(score.shape[0]):
            s = score[i] #[1,128,128]
            g = geometry[i]  #[5,128,128]

            boxes = utils.get_boxes(score_map=s, geo_map=g)
            boxes = boxes if boxes is not None else[[0] * 9] # mask none
            pred_boxes.append(boxes)
        
        pred_boxes = np.array(pred_boxes)
        if pred_boxes.all() == 0:
            return score_map, geo_map
        
        rois = self.roiratate(feature_map, pred_boxes[:,:, :8]) #[b,n,8]
            
            

        return rois