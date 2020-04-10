import torch
import yaml
from dataset import SynthTextDataset
from model_new import CRNN
import numpy as np
from utils import collate_pad, decode, convert2str#, decode_new
from torch.autograd import Variable

# config
config_file = open('config.yaml', 'r', encoding='utf-8')
config = config_file.read()
config_file.close()
config = yaml.full_load(config)

# dict
lexicon = [x for x in config['lexicon']['chars']]

# dataset
train_dataset = SynthTextDataset(config, mode='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=32,
                                                shuffle=True,
                                                drop_last=False,
                                                num_workers=0,
                                                collate_fn=collate_pad)

val_dataset = SynthTextDataset(config, mode='val')
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=32,
                                            shuffle=True,
                                            drop_last=False,
                                            num_workers=0,
                                            collate_fn=collate_pad)
# model
imgH = config['crnn']['imgH']
nc = config['crnn']['nc']
nClass = config['crnn']['nClass']
nh = config['crnn']['nh']
blank_index = config['lexicon']['blank']

model = CRNN(nClass)

model.eval()
for b, (img, label, label_encode) in enumerate(val_dataloader):

    output = model.forward(img)
    
    decoded = decode(output.detach().numpy())  # [batch, max_length]
    res = convert2str(decoded, lexicon)
    print(decoded, label)
    print(res)
    break
