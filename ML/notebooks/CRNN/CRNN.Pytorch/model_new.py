import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):

    def __init__(self, nClass):
        super(CRNN, self).__init__()

        self.conv1 = nn.Conv2d(1  , 256, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(256, 256, (3, 3), (2, 1), (1, 1))
        self.conv6 = nn.Conv2d(256, 256, (3, 3), (2, 1), (1, 1))
        self.avg_pool = nn.AvgPool2d((8, 1))

        self.rnn = nn.LSTM(256, 256, bidirectional=True)
        self.fc = nn.Linear(512, nClass)

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, inputs):
        '''inputs height 32, channel 1'''
        x = self.conv1(inputs)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv5(x)  # [b,c,16,w]
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv6(x)  # [b,c,8,w]
        x = F.relu(x)

        x = self.avg_pool(x)  # [b,c,1,w]
        
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)  # [w,b,256]
        x, _ = self.rnn(x)

        t, b, c = x.size()
        x = x.reshape((t * b, c))

        output = self.fc(x)
        output = output.reshape((t, b, -1))  # [t,b,nClass]

        output = F.log_softmax(output, dim=2)

        return output

if __name__ == "__main__":
    model = CRNN(63)

    x = torch.randn((2, 1, 32, 100))
    y = model(x)
    print(y.shape)
    
