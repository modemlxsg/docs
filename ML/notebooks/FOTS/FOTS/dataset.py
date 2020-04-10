import torch
import torch.nn as nn
import numpy as np
import os
import glob
from cv2 import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from shapely.geometry import Polygon, Point, LineString
import pyclipper

class ICDAR2015_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.root_dir = "H:/dataset/ICDAR2015"
        self.image_dir = os.path.join(self.root_dir, "train_images")
        self.gt_dir = os.path.join(self.root_dir, "train_gts")
        self.imgs = glob.glob(os.path.join(self.image_dir,"*.jpg"))
        self.gts = glob.glob(os.path.join(self.gt_dir, "*.txt"))
        self.imgs.sort(key=lambda x: int(x.split('.')[0].split('_')[2]))
        self.gts.sort(key=lambda x: int(x.split('.')[0].split('_')[3]))

        self.aug = iaa.Sequential([
            iaa.PadToSquare(),
            iaa.Resize(512),
            # iaa.Affine(rotate=(-10, 10))
        ])

        self.datautil = DataUtil_FOTS()

    def __getitem__(self, index):
        img, boxes, label = self.readData(index)
        img_aug, boxes_aug, label_aug = self.augmenter(img, boxes, label)

        score_map, geo_map, train_mask = self.datautil.generate_rbox(img_aug, boxes_aug)

        return img_aug, score_map, geo_map, train_mask
        
    def __len__(self):
        return len(self.imgs)
        
    
    def augmenter(self, img, boxes, label):
        box_reshape = boxes.reshape((-1,2))
        keypoints = ia.KeypointsOnImage.from_xy_array(box_reshape,shape=img.shape)
        img_aug, kps_aug = self.aug.augment(image=img,keypoints=keypoints)
        new_boxes = kps_aug.to_xy_array().reshape(-1, 8)
        
        return img_aug, new_boxes, label
        
    def readData(self,index): # (720, 1280, 3) (7, 8) (7, 1)
        img_path = self.imgs[index]
        gt_path = self.gts[index]

        img = cv2.imread(img_path)
        img = np.array(img)
        img = img[:,:,::-1] #bgr->rgb

        boxes = []
        labels = []
        with open(self.gts[index], encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                gt = line.lstrip('\ufeff').rstrip('\n').split(',')
                label = ",".join(gt[8:])
                box = gt[:8]
                boxes.append(box)
                labels.append(label)

        boxes = np.array(boxes)
        labels = np.array(labels)

        return img, boxes, labels 


class DataUtil_FOTS:

    def __init__(self):
        self.shrink_ratio = 0.4

    def generate_rbox(self, img, boxes,labels):
        score_map, train_mask, index_map = self.generate_score_map(img, boxes,labels)
        geo_map = self.generate_geo_map(img, index_map, train_mask)
        
        return score_map, train_mask, geo_map

    def generate_score_map(self, img, boxes, labels):
        score_map = np.zeros((img.shape[0],img.shape[1]), dtype=np.float32)
        train_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
        index_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        for i,box in enumerate(boxes):
            poly = box.reshape(-1,2)
            height = max(poly[:, 1]) - min(poly[:, 1])
            width = max(poly[:, 0]) - min(poly[:, 0])
            polygon = Polygon(poly)

            # score_map

            if labels[i] == '###':
                cv2.fillPoly(train_mask, [poly.astype(np.int32)], 0)
                continue
            else:
                distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                subject = poly
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)

                if len(shrinked) == 0:
                    cv2.fillPoly(train_mask, [poly.astype(np.int32)], 0)
                    continue
                else:
                    shrinked = np.array(shrinked[0]).reshape(-1, 2)
                    if shrinked.shape[0] > 2 and Polygon(shrinked).is_valid:
                        cv2.fillPoly(score_map, [shrinked.astype(np.int32)], 1)
                        cv2.fillPoly(index_map, [shrinked.astype(np.int32)], i+1)
                    else:
                        cv2.fillPoly(train_mask, [poly.astype(np.int32)], 0)
                        continue
            
        return score_map, train_mask, index_map

    def generate_geo_map(self, img, boxes, index_map, train_mask):
        geo_map = np.zeros((img.shape[0], img.shape[1], 5), dtype=np.float32)
        # geo_map
        for i, box in enumerate(boxes):
            poly = box.reshape(-1,2)
            index_map = index_map * train_mask
            xy = np.argwhere(index_map == i+1)
            if xy.size == 0:
                continue

            xy = xy[:,::-1]  #n*2
            rotate_rect = cv2.minAreaRect(poly)
            box_w, box_h, angle = rotate_rect[1][0], rotate_rect[1][1], abs(rotate_rect[2])
            rect = cv2.boxPoints(rotate_rect)  # 4*2
            
            # sort rect
            p_lowest = np.argmax(rect[:, 1])  #lowest index
            p_lowest_count = np.count_nonzero(rect[:, 1] == rect[p_lowest, 1])
            if p_lowest_count == 2:
                p_0 = np.argmin(np.sum(rect, axis = 1))
                p_1 = (p_0 + 1) % 4
                p_2 = (p_0 + 2) % 4
                p_3 = (p_0 + 3) % 4
            elif angle <= 45 :
                p_3 = p_lowest
                p_0 = (p_lowest + 1) % 4
                p_1 = (p_lowest + 2) % 4
                p_2 = (p_lowest + 3) % 4
            elif angle > 45:
                angle = angle - 90
                p_2 = p_lowest
                p_3 = (p_lowest + 1) % 4
                p_0 = (p_lowest + 2) % 4
                p_1 = (p_lowest + 3) % 4
            rect = rect[[p_0, p_1, p_2, p_3]]
            print(f"angle: {angle}")
            # calculate_distance
            for x, y in xy:
                geo_map[y, x, 0] = self.calculate_distance(x, y, rect[0], rect[1])
                geo_map[y, x, 1] = self.calculate_distance(x, y, rect[1], rect[2])
                geo_map[y, x, 2] = self.calculate_distance(x, y, rect[2], rect[3])
                geo_map[y, x, 3] = self.calculate_distance(x, y, rect[3], rect[0])
                geo_map[y, x, 4] = angle
            
        return geo_map
    
    def calculate_distance(self, x, y, point_1, point_2):
        return Point(x,y).distance(LineString([point_1,point_2]))


            
        
        