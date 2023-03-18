'''

Aum Sri Sai Ram

ABAW-5 for Expr, 2023

MutexMatch with threshold EfficientNet backbone

'''

import torch.nn as nn

class ReverseCLS(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ReverseCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.main = nn.Sequential(self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out


class EfficientNet(nn.Module):
    def __init__(self, num_classes, base_net, ver = 'b0'):
        super(EfficientNet, self).__init__() 
        
        backbone = list(base_net.children())[:-1]
        if ver == 'b0':
            backbone.extend([nn.AdaptiveAvgPool2d(output_size=1), nn.Flatten(), nn.Dropout(p=0.2, inplace=False)])
        elif ver == 'b4':
            backbone.extend([nn.AdaptiveAvgPool2d(output_size=1), nn.Flatten(), nn.Dropout(p=0.4, inplace=False)])
        
        self.backbone = nn.Sequential(*backbone)
        # self.fc = nn.Linear(1280,num_classes)
        self.__in_features=base_net.classifier.fc.in_features
        self.classifier = nn.Linear(self.__in_features, num_classes)

    def forward(self, x, ood_test=False):
        # print(f'In resnet x: {x.shape} ')
        feature = self.backbone(x).squeeze()
        # print(f'In resnet feature: {feature.shape} ')
        output = self.classifier(feature)

        if ood_test:
            return output, feature
        else:
            return output
        
    def output_num(self):
        return self.__in_features

class build_EfficientNetb0:
    def __init__(self, base_net = None):
        # pass
        self.base_net = base_net
    
    def build(self, num_classes):
        return EfficientNet(num_classes = num_classes, base_net = self.base_net, ver = 'b0') 

class build_EfficientNetb4:
    def __init__(self, base_net = None):
        # pass
        self.base_net = base_net
    
    def build(self, num_classes):
        return EfficientNet(num_classes = num_classes, base_net = self.base_net, ver = 'b4') 
    
if __name__ == '__main__':
    pass