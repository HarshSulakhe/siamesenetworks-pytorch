import torch.nn as nn
import torch


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
        nn.Conv2d(3,32,3),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.MaxPool2d(2,2),
        # Currently 45x55x32
        nn.Conv2d(32,64,3),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.MaxPool2d(2,2),
        #Currently 21x26x64
        nn.Conv2d(64,64,3),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.MaxPool2d(2,2),
        #Currently 9x12x64
        )

        self.fc1 = nn.Sequential(
        nn.Linear(6912 , 4096),
        nn.Sigmoid(),
        nn.Dropout(0.5,False),
        nn.Linear(4096,128))
        # self.out = nn.Linear(4096,1)

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(-1,6912)
        output = self.fc1(output)
        return output

    def forward(self, input1,input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # out = self.out(torch.abs(output1-output2))
        # return out.view(out.size())
        return output1,output2
