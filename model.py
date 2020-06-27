import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
        nn.Conv2d(3,64,5),
        nn.ReLU(True),
        nn.MaxPool2d(2,2),
        # Currently 110x110x64
        nn.Conv2d(64,96,3),
        nn.ReLU(True),
        nn.MaxPool2d(2,2),
        #Currently 54x54x96
        nn.Conv2d(96,128,3),
        nn.ReLU(True),
        nn.MaxPool2d(2,2),
        #Currently 26x26x128
        nn.Conv2d(128,128,3),
        nn.ReLU(True),
        nn.MaxPool2d(2,2)
        #Currently 12x12x128
        )

        self.fc1 = nn.Sequential(
        nn.Linear(18432 , 1024),
        nn.Dropout(),
        nn.Linear(1024,128))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(-1,18432)
        output = self.fc1(output)
        return output

    def forward(self, input1):
        output1 = self.forward_once(input1)
        # output2 = self.forward_once(input2)
        return output1
