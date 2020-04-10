import sys
import os
sys.path.append(os.getcwd())

import pytest
from FOTS.model import *
from FOTS.dataset import *
from FOTS.utils import *


def test_shareconv():
    net = SharedConv()
    x = torch.randn(size=(1, 3, 512, 512))
    out = net(x)
    print(out.shape)
    assert out.shape == (1, 32, 128, 128)
    
    
def test_detector():
    net = Detector()
    x = torch.randn(size=(1, 32, 128, 128))
    score, geo = net(x)

    assert score.shape == (1, 1, 128, 128)
    assert geo.shape == (1, 5, 128, 128)
    
def test_east():
    net = EAST()
    x = torch.randn(size=(8, 3, 512, 512))
    o = net.forward(x)


def test_dataset():
    dataset = ICDAR2015_Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    
    data = iter(dataloader)
    batch_data = next(data)

    img, score_map, geo_map = batch_data
    print(score_map.shape, geo_map.shape)
    
def test_datautil():
    dataset = ICDAR2015_Dataset()
    img, boxes, labels  = dataset.readData(1)
    img_aug, boxes_aug, labels_aug = dataset.augmenter(img, boxes, labels)
    
    data_util = DataUtil_FOTS()
    score_map, train_mask, index_map = data_util.generate_score_map(img_aug, boxes_aug, labels_aug)
    geo_map = data_util.generate_geo_map(img_aug, boxes_aug, index_map, train_mask)

    score_map = score_map * train_mask
    score_map = score_map[np.newaxis,:,:]
    geo_map = geo_map.transpose(2, 0, 1)
    
    boxes = get_boxes_new(score_map, geo_map)
    print(boxes.shape)


    

    



