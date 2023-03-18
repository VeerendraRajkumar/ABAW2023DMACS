# -*- coding:utf-8 -*-
'''
Aum Sri Sai Ram

ABAW-5 for Expr, 2023

Noise aware model - main function

'''

import os
import cv2
import time
import torch
import argparse
import torch.utils.data as data
import torchvision.transforms as transforms

from algorithm.noisyfer_aug import noisyfer
from PIL import Image

from algorithm import transform as T
from algorithm.randaugment import RandAugmentMC

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
parser.add_argument('--aff_path', type=str, default='../../data/ABAW23/Aff_wild2/cropped_aligned', help='Raf-DB dataset path.')

parser.add_argument('--pretrained', type=str, default='pretrained/epoch_5_noise_affectnet8_806-12-2022-10-22-59train_affectnet8_fullpath_list.txt_classes_8__affectnet7__acc_60.47.pth',  help='Pretrained weights')
parser.add_argument('--comment', type=str, default="usingwce_flippedimagestrongaugment_nosteplr(mon mar 13)",help="")
parser.add_argument('--resume', type=str, default='chkpt/aff/affectnet/epoch_13_f1_0.3346.pth', help='resume from saved model')
                                         
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
                    
parser.add_argument('--dataset', type=str, help='rafdb, ferplus, affectnet', default='aff')
parser.add_argument('--noise_file', type=str, help='', default='')
parser.add_argument('--beta', type=float, default=0.25,  help='..based on ')
parser.add_argument('--alpha', type=float, default=0.5,  help='..based on ')
parser.add_argument('--eps', type=float, default=0.35,  help='..based on ')
parser.add_argument('--co_lambda_max', type=float, default=.9,   help='..based on ')
parser.add_argument('--n_epoch', type=int, default=40)
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=20)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--batch_size', type=int, default=150, help='batch_size')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--gpu', default='2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--adjust_lr', type=int, default=1)
parser.add_argument('--num_models', type=int, default=1)
parser.add_argument('--relabel_epochs', type=int, default=40)
parser.add_argument('--warmup_epochs', type=int, default=1)
parser.add_argument('--log_file', type=str, default="log/aff/affectnet",help="feb/raf/30--for february rafdb 30%noise")
parser.add_argument('--cp_file', type=str, default="chkpt/aff/affectnet")
parser.add_argument('--model_type', type=str, help='[mlp,cnn,res]', default='res')
parser.add_argument('--w', type=int, default=7, help='width of the attention map')
parser.add_argument('--h', type=int, default=7, help='height of the attention map')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print("arguments state :",state)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# Seed
torch.manual_seed(args.seed)
if args.gpu is not None:
    torch.cuda.manual_seed(args.seed)

else:
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

# Hyper Parameters
batch_size = args.batch_size
learning_rate = args.lr

def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            img = Image.fromarray(img)
            return img
    except IOError:
        print('Cannot load image ' + path)


class Dataset_ABAW_Affwild2(torch.utils.data.Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader):

        self.root = root
        self.transform = transform
        self.loader = loader

        self.image_list = []
        self.labels_Exp =  []
        
        with open(file_list) as f:
            img_label_list = f.read().splitlines()
            
        for info in img_label_list:#[:1000]:            
            details = info.split(' ')
            if details[1] == '-1' :
               continue 
            self.labels_Exp.append(int(details[1]))
            self.image_list.append(details[0])
        print('Total samples: ', len(self.image_list))   
        
    def __getitem__(self, index):

        img_path = self.image_list[index]
        img = self.loader(os.path.join(self.root, img_path))
        flip_image = img.transpose(Image.FLIP_LEFT_RIGHT)
       
        if self.transform is not None:
            img1 = self.transform[0](img)
            img2 = self.transform[1](RandAugmentMC(2, 10)(flip_image))
        label_exp = int(self.labels_Exp[index])
        
        
        return img1,img2, label_exp,  img_path
    
    def change_emotion_label_same_as_rafdb(self, emo_to_return):
        
        dict_class_names = {0:6,1:5,2:2,3:1,4:3,5:4,6:0,7:7}
        return dict_class_names[emo_to_return]
    
    def __len__(self):
        return len(self.image_list)
                         

def main():
    
    print('\n\t\t\tAum Sri Sai Ram\n\n\n')
    
    t=time.localtime()
    time_stamp=time.strftime('%d-%m-%Y-%H-%M-%S',t)
    
    #alpha = str(args.alpha)  
    if(args.log_file):
        txtfile = args.log_file+'/log_'+args.dataset+'__'+time_stamp+"__"+args.noise_file.split('/')[-1]+"_warmup_"+str(args.warmup_epochs)+".txt"
    else:
        txtfile = "temp.txt"   
    
    if  args.dataset == 'aff':   
        input_channel = 3
        num_classes = args.num_classes
        
        args.epoch_decay_start = 100    
        
        args.model_type = "res"
        
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        
        
        trans_weak = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])
        train_transforms = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            #transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.25)) ])
        
        
        data_transforms_val = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
                                     
        train_transforms = [trans_weak,train_transforms]
        test_transforms = [data_transforms_val,data_transforms_val]
        train_dataset = Dataset_ABAW_Affwild2(root="/home/darshan/data/ABAW23/Aff_wild2/cropped_aligned",file_list ="/home/darshan/data/ABAW23/Aff_wild2/valid_training_set_annotations_23.txt",transform=train_transforms)
        print('Train set size:', train_dataset.__len__())  
        test_dataset =  Dataset_ABAW_Affwild2(root="/home/darshan/data/ABAW23/Aff_wild2/cropped_aligned",file_list ="/home/darshan/data/ABAW23/Aff_wild2/valid_validation_set_annotations_23.txt",transform=test_transforms)
        print('Validation set size:', test_dataset.__len__())
        
    else:
        print('Invalid dataset')
        
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               num_workers = args.num_workers,
                                               drop_last=True,
                                               shuffle = True,  
                                               pin_memory = True) 
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = 5*batch_size,
                                               num_workers = args.num_workers,
                                               shuffle = False,  
                                               pin_memory = True)    
                                               
                                                                            
    
    model= noisyfer(args, train_dataset,  input_channel, num_classes)
    
    with open(txtfile, "a") as myfile:
        myfile.write('epoch train_acc   test_acc\n')
    
    best_f1   = 0.0   
    # training
    continue_epoch =0
    if(args.resume):
        continue_epoch = int(args.resume.split('_')[1]) + 1
    best_epoch=0
    train_acc =0.0
    for epoch in range(continue_epoch, args.n_epoch):
        start= time.time()
        train_acc = model.train(train_loader, epoch)
        #train_time = time.time() - start
        test_acc,f1 =  model.evaluate(test_loader)
        end = time.time() - start
        
        
        if best_f1 <   f1:
            best_f1 = f1
            best_epoch=epoch+1
            if(args.cp_file):
                model.save_model(epoch, f1, args.noise_file.split('/')[-1],args)  
            
             
                
        print(  'Epoch [%d/%d] Test Accuracy on the %s test images: Accuracy %.4f f1_score %.4f'  % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc,f1))
        
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)+1) + '  '  + str(train_acc) +'  '  + str(test_acc) +"\n")
                
        print(f"\ntime_taken for training epoch : {epoch+1}  is {end/60}min ")
        
        

    print('\n\n \t Best Test acc for {} at epoch {} is {}: '.format(args.noise_file,best_epoch, best_f1))    


if __name__ == '__main__':
   
    main()    

