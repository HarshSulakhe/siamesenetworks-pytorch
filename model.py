import torch.nn as nn


class SiameseNetwork(nn.Module):
     def __init__(self):
          super(SiameseNetwork, self).__init__()
          # Setting up the Sequential of CNN Layers
          self.cnn1 = nn.Sequential(
          nn.Conv2d(1, 96, kernel_size=11,stride=1),
          nn.ReLU(inplace=True),
          nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
          nn.MaxPool2d(3, stride=2),

          nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
          nn.ReLU(inplace=True),
          nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
          nn.MaxPool2d(3, stride=2),
          nn.Dropout2d(p=0.3),

          nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(3, stride=2),
          nn.Dropout2d(p=0.3),
          )

          # Defining the fully connected layers
          self.fc1 = nn.Sequential(
          # First Dense Layer
          nn.Linear(30976, 1024),
          nn.ReLU(inplace=True),
          nn.Dropout2d(p=0.5),
          # Second Dense Layer
          nn.Linear(1024, 128),
          nn.ReLU(inplace=True),
          # Final Dense Layer
          nn.Linear(128,2))

     def forward_once(self, x):
          # Forward pass
          output = self.cnn1(x)
          output = output.view(output.size()[0], -1)
          output = self.fc1(output)
          return output

    def forward(self, input1, input2):
         # forward pass of input 1
         output1 = self.forward_once(input1)
         # forward pass of input 2
         output2 = self.forward_once(input2)
         # returning the feature vectors of two inputs
         return output1, output2
