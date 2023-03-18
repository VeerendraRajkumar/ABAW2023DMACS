'''
Aum Sri Sai Ram

ABAW-5 for Expr, 2023

Noise aware model - making renet model

'''
import torch
from model.resnet import *


def resModel(args): #resnet18

        """
        This function gives back the model where the backbone is resnet18 with MSCeleb-1M weights are loaded into it.
        The resnet18 function is tweaked in its forward function so that we get both the predictions and class activation maps.
        
        """
   
        model = torch.nn.DataParallel(resnet18(num_classes=args.num_classes,end2end= False,pretrained= False)).cuda()
    
        if args.pretrained:
       
            checkpoint = torch.load(args.pretrained)
            print(f"loading checkpoint from {args.pretrained}")
            #pretrained_state_dict = checkpoint['state_dict']
            pretrained_state_dict = checkpoint['model']
            model_state_dict = model.state_dict()
         
            for key in pretrained_state_dict:
                if  ((key == 'module.fc.weight') | (key=='module.fc.bias') | (key=='module.feature.weight') | (key=='module.feature.bias') ) :
                    pass
                else:
                    model_state_dict[key] = pretrained_state_dict[key]

            model.load_state_dict(model_state_dict, strict = True)
            print('Model loaded from Msceleb pretrained')


        else:
            print('No pretrained resent18 model built.')
        return model 

    

