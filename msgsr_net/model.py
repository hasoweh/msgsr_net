import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18


class CamGenerationUnit(nn.Module):
    def __init__(self, n_feat, n_class):
        super().__init__()

        self.conv = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(n_feat)
        self.relu = nn.ReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(n_feat, n_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x

class MSGSRNet(nn.Module):
    def __init__(self, backbone, n_class, n_bands, return_branch=None):
        """
        Parameters
        ----------
        backbone : A CNN backbone feature extraction network
        n_class : int, number of classes to predict from
        n_bands : int, number of bands in the input images
        return_branch : None or int
            If None, then the network will return a list of all logits from
            each CG Unit as the output from the forward pass. This should be
            used during training.
            If an int, must be from 0-3, and will return only the logits from
            the given layer as the output from the forward pass. This should be
            used when running inference to generate CAMs at different levels of
            the network.
            
        NOTE
        ----
        Currently only written to support RESNET variants. If you wish to use a 
        different backbone you will need to edit the forward function.
        """
        super().__init__()
        self.cg0 = CamGenerationUnit(64, n_class)
        self.cg1 = CamGenerationUnit(128, n_class)
        self.cg2 = CamGenerationUnit(256, n_class)
        self.cg3 = CamGenerationUnit(512, n_class)
        self.model = backbone
        
        # determines if we want the output from a specific CGU
        # this is relevant when we do the GradCAM++ calc
        self.return_branch = return_branch
        
        if n_bands != 3:
            self.model.conv1 = nn.Conv2d(n_bands, 64, kernel_size=7, 
                                         stride=2, padding=3,
                                         bias=False)

    def forward(self, x):
        logits = []
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        logits.append(self.cg0(x))
        x = self.model.layer2(x)
        logits.append(self.cg1(x))
        x = self.model.layer3(x)
        logits.append(self.cg2(x))
        x = self.model.layer4(x)
        logits.append(self.cg3(x))

        if self.return_branch:
            return logits[self.return_branch]
        else:
            return logits