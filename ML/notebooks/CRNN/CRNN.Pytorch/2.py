import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import cv2
from dataset import SynthTextDataset
import yaml

config_file = open('config.yaml', 'r', encoding='utf-8')
config = config_file.read()
config_file.close()
config = yaml.full_load(config)

train_dataset = SynthTextDataset(config, mode='train')

img, label, le = train_dataset.__getitem__(0)

print(img.shape)
