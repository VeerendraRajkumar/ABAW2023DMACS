'''
Aum Sri Sai Ram

ABAW-5 for Expr, 2023

using resnet18, using only valid-labeled images
Fully Supervised
 
'''
# Import Essentials
from __future__ import print_function

import os
import time
import pytz
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Import essential libraries from torch
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# import required user defined libraries
from models.backbone import ResNet_18
import dataset.abaw_baseline as dataset
from losses import cross_entropy_loss_without_weights
from utils import Logger, AverageMeter, accuracy, mkdir_p, EXPR_metric

# Parser
parser = argparse.ArgumentParser(description='Supervised Training')
# Optimization options
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
                        
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=5, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0,1,2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
                    
parser.add_argument('--train-iteration', type=int, default=800,help='Number of iteration per epoch')
                    
parser.add_argument('--out', default='result',  help='Directory to output the result')

parser.add_argument('--model-dir','-m', default='baseline', type=str)

#Data
parser.add_argument('--train-root', type=str, default='/home/darshan/data/ABAW23/Aff_wild2/cropped_aligned',
                        help="root path to train data directory")
parser.add_argument('--val_root', type=str, default='/home/darshan/data/ABAW23/Aff_wild2/cropped_aligned',
                        help="root path to test data directory")
                        
parser.add_argument('--label-train', default='/home/darshan/data/ABAW23/Aff_wild2/valid_training_set_annotations_23.txt', type=str, help='')
parser.add_argument('--label-val', default='/home/darshan/data/ABAW23/Aff_wild2/valid_validation_set_annotations_23.txt', type=str, help='')

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

best_perf = 0  

def main():
    global best_perf

    # check and make missing directories
    if not os.path.isdir(args.out):
        mkdir_p(args.out)
    if not os.path.isdir(args.model_dir):
        mkdir_p(args.model_dir)

    # Data
    print(f'==> Preparing DB')
    # mean and std to normalize images
    # from ImageNet
    mean=[0.485, 0.456, 0.406]
    std =[0.229, 0.224, 0.225]

    # Transformations for training
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomApply([
            transforms.RandomCrop(224, padding=8)
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Transformations for validation
    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # getting datasets
    train_set = dataset.Dataset_ABAW_Affwild2(args.train_root, args.label_train, transform=dataset.TransformTwice(transform_train))
    val_set = dataset.Dataset_ABAW_Affwild2(args.val_root, args.label_val, transform=transform_val)
    
    print('Train dataset size: ', len(train_set))
    print('Val dataset size: ', len(val_set))
    
    # getting data loaders
    trainloader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    
    # Model
    print("==> creating ResNet-18")

    def create_model(ema=False):
        model = ResNet_18()
        model = torch.nn.DataParallel(model).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))  
    
    cudnn.benchmark = True
        
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # setting logger
    logger = Logger(os.path.join(args.out, 'Baseline'.format(datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d-%m-%Y_%H-%M-%S'))), title='ABAWAffwild2')
    logger.set_names(['Train Loss', 'Train acc', 'Val Loss', 'Val Acc.', 'Val F1-Exp'])

    val_perf = []
    threshold = 0.8
    start_epoch = 1
    
    exp_criterion = cross_entropy_loss_without_weights

    # Train and val
    for epoch in range(start_epoch, args.epochs + 1):

        print('\nEpoch: [%d | %d] LR: %f ' % (epoch, args.epochs, state['lr']))        
        train_loss, train_acc, exp_f1  = train(trainloader, model, optimizer, exp_criterion, threshold, epoch, use_cuda)
        val_loss, val_acc,exp_f1, _,_ = validate(val_loader, model, exp_criterion,  epoch, use_cuda, mode='Validation Stats')
        overall_perf = exp_f1
        
        # append logger file
        logger.append([train_loss, train_acc, val_loss, val_acc,exp_f1])
        
        # save model
        is_best = overall_perf > best_perf
        best_perf = max(overall_perf, best_perf)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),                
                'acc': val_acc,
                'best_perf': best_perf,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        val_perf.append(overall_perf)
        
    logger.close()
      
    print('Best Performance:')
    print(best_perf)


def save_checkpoint(state, is_best):
    if is_best:
        torch.save(state, os.path.join(args.model_dir, 'supervised_ep{}_{}.pth.tar'.format(state['epoch'] - 1, state['best_perf'])))
 
def train(trainloader, model, optimizer, exp_criterion, threshold, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    
    args.train_iteration = int(len(trainloader.dataset)/args.batch_size)

    bar = tqdm(range(args.train_iteration))   
    labeled_train_iter = iter(trainloader)
    
    model.train()
    outputs_exp = torch.ones(1).cuda()
    targets_exp = torch.ones(1).long().cuda()
    
    for batch_idx in range(args.train_iteration):
        try:
            (inputs, inputs_strong), label_EXP, img_path = labeled_train_iter.next() 
            
        except:
            labeled_train_iter = iter(trainloader)
            (inputs, inputs_strong), label_EXP, img_path = labeled_train_iter.next()

        
        # measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs.size(0)

        if use_cuda:
            inputs, inputs_strong, label_EXP = inputs.cuda(), inputs_strong.cuda(), label_EXP.cuda()
            
        outputs = model(inputs)

        #print(inputs.shape, outputs.shape, label_V.shape, label_A.shape, label_EXP.shape, label_AU.shape)

        
        # measure performance and record loss
        predicted, target = outputs, label_EXP
        
        loss = exp_criterion(predicted, target)
        loss = loss.mean()
        prec1, prec5 = accuracy(predicted, target, topk=(1, 5))
        
        top1.update(prec1.item(), inputs.size(0))
        outputs_exp = torch.cat((outputs_exp, F.softmax(predicted , dim=-1).argmax(-1).float()), dim=0)
        targets_exp = torch.cat((targets_exp, target), dim=0)
          
        losses.update(loss.item(), inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.set_description(" Loss:{loss:.4f}| Acc:{top1: .4f}".format(
                loss = losses.avg,
                top1=top1.avg,
        ))
        bar.update()
    bar.close()

    exp_f1 = EXPR_metric(outputs_exp.cpu().numpy() , targets_exp.cpu().numpy())    
    print('exp performance',exp_f1)

    return losses.avg, top1.avg, exp_f1     
    
    
def validate(dataloader, model, exp_criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    
    args.validate_iteration = int(len(dataloader.dataset)/args.batch_size)
    bar = tqdm(range(args.validate_iteration))
    labeled_train_iter = iter(dataloader)
    model.eval()
    
    outputs_exp = torch.ones(1).cuda()
    outputs_exp_new = torch.ones(1, args.num_exp_classes).cuda()
    targets_exp = torch.ones(1).long().cuda()
        
    with torch.no_grad():
        for batch_idx in range(args.validate_iteration):
            try:
                inputs, label_EXP, img_path = labeled_train_iter.next() 
                
            except:
                labeled_train_iter = iter(dataloader)
                inputs, label_EXP, img_path = labeled_train_iter.next() 
            
            # measure data loading time
            data_time.update(time.time() - end)
            if mode == 'Train Stats':
                inputs = inputs[0]

            if use_cuda:
                inputs, label_EXP = inputs.cuda(), label_EXP.cuda()

                
            outputs = model(inputs)

            # measure performance and record loss
            predicted, target = outputs, label_EXP
            loss = exp_criterion(predicted, target)
            loss = loss.mean()
            prec1, prec5 = accuracy(predicted, target, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            outputs_exp = torch.cat((outputs_exp,  F.softmax(predicted , dim=-1).argmax(-1).float()), dim=0)
            targets_exp = torch.cat((targets_exp, target), dim=0)
         
            losses.update(loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.set_description(" Loss:{loss:.4f}| Acc:{top1: .4f}".format(
                loss = losses.avg,
                top1=top1.avg,))
            bar.update()
        bar.close()

    exp_f1 = EXPR_metric(outputs_exp.cpu().numpy() , targets_exp.cpu().numpy())
    print('exp performance',exp_f1)
      
    return losses.avg, top1.avg, exp_f1, outputs_exp_new, targets_exp
                    
if __name__ == '__main__':
    main()                        

