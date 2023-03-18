"""
Aum Sri Sai Ram

ABAW-5 ,2023

TEST using resnet18 supervised

Code to get predictions on test set for supervised model.

"""

from __future__ import print_function

import os
import time
import pytz
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.parallel
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
 
from models.backbone import ResNet_18
import dataset.abaw_baseline as dataset
from utils import  AverageMeter, mkdir_p


# Parser
parser = argparse.ArgumentParser(description='Test Code')
# Optimization options

parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')

# Miscs
parser.add_argument('--manualSeed', type=int, default=5, help='manual seed')
#Device options
parser.add_argument('--gpu', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
                    
parser.add_argument('--out', default='preds',  help='Directory to output the predictions')

#Data
parser.add_argument('--test-root', type=str, default='/home/darshan/data/ABAW23/Aff_wild2/cropped_aligned',
                        help="root path to train data directory")
                        
parser.add_argument('--label-test', default='/home/darshan/data/ABAW23/Aff_wild2/test/test_list_23.txt', type=str, help='')

parser.add_argument('--chkpath', default='', type=str, help='path to checkpoint on which to run the test set')

parser.add_argument('--num_exp_classes', type=int, default=8, help='number of expression classes')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

def main():

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing DB')
    mean=[0.485, 0.456, 0.406]
    std =[0.229, 0.224, 0.225]
	
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    test_set = dataset.Dataset_ABAW_test(args.test_root, args.label_test, transform=transform_test)

    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print('Test dataset size: ', len(test_set))
    
    # Model
    print("==> creating ResNet-18")

    def create_model(ema=False):
        model = ResNet_18()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.chkpath)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    cudnn.benchmark = True
    model = create_model()
    
        
    #Validating the model
    test(test_loader, model, use_cuda, mode='Test Stats')#3 <- (4)


def test(dataloader, model, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    
    args.validate_iteration = int(len(dataloader.dataset)/args.batch_size)

    outpath = (os.path.join(args.out, 'Preds_baseline_{}'.format(datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d-%m-%Y_%H-%M-%S'))))
    fout = open(outpath, 'w')
    print('image_location,Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other', file=fout)
    
    bar = tqdm(range(args.validate_iteration))
    model.eval()
         
    with torch.no_grad():
        for batch_idx, (img, name) in enumerate(dataloader):

            # measure data loading time
            data_time.update(time.time() - end)

            batch_size = img.size(0)
            #print('batch_size : ', batch_size)
            
            if use_cuda:
               inputs = img.cuda()
               
            outputs = model(inputs)
            
            outputs_exp = F.softmax(outputs, dim=-1).argmax(-1).float()
            
            for image in range(batch_size):
                out_exp = str(int(outputs_exp[image]))
                s = (name[image] + ','+ out_exp)	
                print(s, file=fout)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # plot progress
            bar.set_description('(({batch}/{size})'.format(
                       batch=batch_idx + 1,
                       size=args.validate_iteration))
            bar.update()
    bar.close()
    fout.close()
    
if __name__ == '__main__':
    main()  
