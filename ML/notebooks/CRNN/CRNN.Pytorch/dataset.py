import torch
import os
import cv2
import numpy as np



class SynthTextDataset(torch.utils.data.Dataset):

    def __init__(self, config, mode='train'):
        super(SynthTextDataset, self).__init__()
        self.root_dir = config['dataset']['SynthText']['root_dir']
        self.item_num = config['dataset']['SynthText'][f'{mode}_num']
        self.imgH = config['dataset']['SynthText']['img_height']
        self.mode = mode
        self.lexicon = [x for x in config['lexicon']['chars']]

        self.item_path = self.readItemPath()
    
    def __getitem__(self, index):
        path = self.item_path[index]
        label = (path.split('/')[-1]).split('_')[1]
        im = cv2.imread(path)[:,:,::-1]
        im = self.augmenter(im)
        label_encode = self.encodeLabel(label)

        return im, label, label_encode

    def __len__(self):
        return len(self.item_path)
    
    def readItemPath(self):
        if self.mode not in ['train', 'test', 'val']:
            raise Exception("mode error")

        with open(os.path.join(self.root_dir, f"annotation_{self.mode}.txt")) as f:
            lines = f.readlines()

        i, items = 0, []
        for line in lines:
            if i >= self.item_num:
                break
            fullpath = os.path.join(self.root_dir, line.split(' ')[0])
            if os.path.exists(fullpath):
                items.append(fullpath)
                i += 1

        return items
        
    def augmenter(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)  # to gray
        # im = cv2.resize(im, (0, 0), fy=self.imgH / im.shape[0], fx=self.imgH / im.shape[0])
        im = cv2.resize(im, (100, 32))
        im = im[:,:, np.newaxis]
        im = im / 255.0
        return im

    def encodeLabel(self, label):

        return np.array([self.lexicon.index(char) for char in label])
        
        