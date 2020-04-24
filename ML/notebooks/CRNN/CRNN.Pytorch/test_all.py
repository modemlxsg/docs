import pytest
import yaml
from dataset import SynthTextDataset
from model import CRNN
import torch


def test_dataset():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        data = f.read()
        config = yaml.full_load(data)
    
    ds = SynthTextDataset(config, mode='train')

    img, label = ds.__getitem__(1)
    print(img.shape, label)


def test_model():
    model = CRNN(32, 1, 53, 256)
    inp = torch.randn((1, 1,32, 100))

    out = model.forward(inp)
    print(out.shape)
    