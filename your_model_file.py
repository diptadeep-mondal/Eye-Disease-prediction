import torch
import torch.nn as nn

class animal_classification(nn.Module):
  def __init__(self):
    super().__init__()
    self.block1=nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.block2=nn.Sequential(
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.layer=nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=256 *32 *32,out_features=4)
    )
  def forward(self,x:torch.tensor):
    return self.layer(self.block2(self.block1(x)))