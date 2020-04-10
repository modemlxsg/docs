import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision



class ShareConv(nn.Module):
    def __init__(self):
        super(ShareConv,self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        
    def forward(self, inputs):
        x = inputs
        
        for name,layer in self.backbone.named_children():
            x = layer(x)
            print(name , x.shape)
            if name == "layer1":
                d4 = x
            elif name == "layer2":
                d8 = x
            elif name == "layer3":
                d16 = x
            elif name == "layer4":
                d32 = x
                break
        
        u1 = F.interpolate(d32, scale_factor=2.0, mode="bilinear", align_corners=True)
        u1 = torch.cat([u1, d16], dim=1)
        
        u2 = F.interpolate(u1, scale_factor=2.0, mode="bilinear", align_corners=True)
        u2 = torch.cat([u2, d8], dim=1)

        u3 = F.interpolate(u2, scale_factor=2.0, mode="bilinear", align_corners=True)
        u3 = torch.cat([u3, d4], dim=1)


        print(u1.shape,u2.shape,u3.shape)


        return inputs


class DBNet(nn.Module):
    def __init__(self):
        super(DBNet,self).__init__()
        

    def forward(self,inputs):
        

        return inputs



if __name__ == "__main__":
    inputs = np.random.random((16,3,640,640))
    inputs = torch.tensor(inputs, dtype=torch.float32)
    model = ShareConv()
    print(model.forward(inputs))