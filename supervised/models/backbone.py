'''
Aum Sri Sai Ram

ABAW-5 for Expr, 2023

backbone for Fully Supervised
 
'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)  

class ResNet_18(nn.Module):
    def __init__(self, num_exp_classes=8, num_au = 12, num_va=2, fc_layer_dim = 128 ):
        super(ResNet_18, self).__init__()
        
        ResNet18 = torchvision.models.resnet18(pretrained=False)
        
        checkpoint = torch.load('./models/resnet18_msceleb.pth')
        ResNet18.load_state_dict(checkpoint['state_dict'], strict=True)

        self.base = nn.Sequential(*list(ResNet18.children())[:-2])
        self.output = nn.Sequential(nn.Dropout(0.5), Flatten())
        features_dim =  512
        self.exp_fc = nn.Sequential(nn.Linear(features_dim, fc_layer_dim), nn.ReLU())
        self.exp_classifier = nn.Linear(fc_layer_dim, num_exp_classes)

    def forward(self, image):
        
        feature_map = self.base(image)
        feature_map = F.avg_pool2d(feature_map, feature_map.size()[2:])        
        feature = self.output(feature_map)
        feature = F.normalize(feature, dim=1)
        outputs =  self.exp_classifier( self.exp_fc(feature) )
        
        return outputs   
        

if __name__=='__main__':

    pass