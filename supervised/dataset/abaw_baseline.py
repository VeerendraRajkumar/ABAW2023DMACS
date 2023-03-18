'''
Aum Sri Sai Ram

ABAW-5 for Expr, 2023

Datasets and dataloaders
 
'''

import os
import cv2
import copy
from PIL import Image

import torch
from .randaugment import RandAugment

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform
        self.strong_transfrom = copy.deepcopy(transform)
        self.strong_transfrom.transforms.insert(0, RandAugment(3,5))

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.strong_transfrom(inp)
        return out1, out2

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
            
        for info in img_label_list: # [:100000]:            
            details = info.split(' ')
            if details[1] == '-1' :
               continue 
              
            self.labels_Exp.append(int(details[1]))
            self.image_list.append(details[0])
            
        print('Total samples: ', len(self.image_list))   

    def __getitem__(self, index):
       
        img_path = self.image_list[index]
        img = self.loader(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)
        label_exp = int(self.labels_Exp[index])

        return img, label_exp,  img_path

    
    def __len__(self):
        return len(self.image_list)

class Dataset_ABAW_test(torch.utils.data.Dataset):
	def __init__(self, root, file_list, transform=None, loader=img_loader):
		self.root = root
		self.transform = transform
		self.loader = loader
		self.image_list = []

		with open(file_list) as f:
			img_label_list = f.read().splitlines()

		for info in img_label_list:
			self.image_list.append(info)

	def __getitem__(self, index):

		img_path = self.image_list[index]
		img = self.loader(os.path.join(self.root, img_path))
		if self.transform is not None:
			img = self.transform(img)
  
		return img, img_path

	def __len__(self):
		return len(self.image_list)

if __name__ == '__main__':
    pass