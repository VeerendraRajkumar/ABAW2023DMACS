'''
Aum Sri Sai Ram

ABAW-5 for Expr, 2023

MutexMatch with threshold - Dataset and data loaders for MutexMatch with threshold

'''

import os
import cv2
import copy
import math
import numpy as np
from PIL import Image

import torch
from .randaugment import RandAugment
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, SequentialSampler

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Data
print('==> Preparing AFF-DB')
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomApply([
        transforms.RandomCrop(224, padding=8)
    ], p=0.125),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform
        self.strong_transfrom = copy.deepcopy(transform)
        self.strong_transfrom.transforms.insert(0, RandAugment(3,5))

    def __call__(self, inp):
        out_w = self.transform(inp)
        out_s = self.strong_transfrom(inp)
        return out_w, out_s

def get_aff(num_classes, n_lab_pc = 195653,
            train_root = '/home/mtech2/Raj/Prj/data/ABAW23/Aff_wild2/cropped_aligned/',
            train_file_list = '/home/mtech2/Raj/Prj/data/ABAW23/Aff_wild2/valid_training_set_annotations_23.txt',
            test_root = '/home/mtech2/Raj/Prj/data/ABAW23/Aff_wild2/cropped_aligned/',
            test_file_list = '/home/mtech2/Raj/Prj/data/ABAW23/Aff_wild2/valid_validation_set_annotations_23.txt', 
            transform_train=None, transform_val=None):
    
    train_labeled_idxs, train_unlabeled_idxs = natural_data_split_full(train_file_list, n_lab_pc, num_classes)
    train_labeled_dataset = Dataset_AFF_labeled(train_root, train_file_list, train_labeled_idxs, transform=transform_train)
    train_unlabeled_dataset = Dataset_AFF_unlabeled(train_root, train_file_list, train_unlabeled_idxs, transform=TransformTwice(transform_train))

    # selecting only valid samples as even in val set there are images with -1 as labels which are to be treated invlaid
    test_idxs = data_split_test(test_file_list)
    test_dataset = Dataset_AFF_labeled(test_root, test_file_list, test_idxs, transform=transform_val)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset
    
def target_read(path):
    " Reads the file at path and returns the labels in that file"
    label_list = []
    with open(path) as f:
        img_label_list = f.read().splitlines()
    for info in img_label_list:
        _, label_name = info.split(' ')
        label_list.append(int(label_name))
    return label_list

def data_split_test(filename):
    labels = target_read(filename)
    labels = np.array(labels) # this is required for np.where to return appropriate indices

    return np.where(labels != -1)[0]

def natural_data_split_full(filename, n_lab_pc=195653, num_classes=8):
    "Performs the split of data into labeled and unlabeled "
    labels = target_read(filename)
    labels = np.array(labels) # this is required for np.where to return appropriate indices
    
    train_unlabeled_idxs = []
    train_labeled_idxs = []

    train_unlabeled_idxs = np.where(labels == -1)[0]
    train_labeled_idxs = np.where(labels != -1)[0]

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs

def natural_data_split(filename, n_lab_pc=195653, num_classes=8,ratio=[0.2997, 0.0262, 0.0181, 0.0202, 0.1500, 0.1211, 0.0506, 0.3142] ):
    """
    Performs the split of data into labeled and unlabeled using the distribution of overall labeled train and validation sets.
    with total labeled images considered  in total only to be n_lab_pc
    """
    
    labels = target_read(filename)
    labels = np.array(labels) # this is required for np.where to return appropriate indices
    
    train_unlabeled_idxs = []
    train_labeled_idxs = []
    
    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:math.ceil(float(n_lab_pc*ratio[i]))+1])
        train_unlabeled_idxs.extend(idxs[math.ceil(float(n_lab_pc*ratio[i]))+1:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs

def data_split(filename, n_lab_pc=9000, num_classes=8):
    "Performs the split of data into labeled and unlabeled with only n_lab_pc images per class"
    labels = target_read(filename)
    labels = np.array(labels) # this is required for np.where to return appropriate indices
    
    train_unlabeled_idxs = []
    train_labeled_idxs = []

    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_lab_pc])
        train_unlabeled_idxs.extend(idxs[n_lab_pc:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs

def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            img = Image.fromarray(img)
            return img
    except IOError:
        print('Cannot load image ' + path)

class Dataset_AFF(torch.utils.data.Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader, type=""):

        self.root = root
        self.transform = transform
        self.loader = loader

        image_list = []
        label_list = []

        with open(file_list) as f:
            img_label_list = f.read().splitlines()

        for info in img_label_list:
            image_path, label_name = info.split(' ')
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]
        
        img = self.loader(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_list)

class Dataset_AFF_labeled(Dataset_AFF):
    def __init__(self, root, file_list, indexs, transform=None):
        super(Dataset_AFF_labeled, self).__init__(root, file_list, transform=transform)

        if indexs is not None:
            self.image_list = np.array(self.image_list)[indexs] 
            self.label_list = np.array(self.label_list)[indexs]

class Dataset_AFF_unlabeled(Dataset_AFF_labeled):
    def __init__(self, root, file_list, indexs, transform=None):
        super(Dataset_AFF_unlabeled, self).__init__(root, file_list, indexs, transform=transform)


def get_data_loaders(dataset = 'aff', num_classes = 8, n_lab_pc=195653, batch_size = 64, mu = 7, iters = 2**20 * 64, num_workers=1):
    """
    return data loaders for train_lab, train_unlab, val/test
    """
    assert dataset.lower().startswith("aff") and num_classes == 8
    
    labeled_dataset, unlabeled_dataset, test_dataset = get_aff(num_classes, n_lab_pc= n_lab_pc, transform_train=transform_train, transform_val=transform_val)
    
    sampler_x = RandomSampler(labeled_dataset, replacement=True, num_samples=iters * batch_size)
    labeled_trainloader = DataLoader(
        labeled_dataset,
        # sampler=batch_sampler_x,
        sampler=sampler_x,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
        )

    sampler_u = RandomSampler(unlabeled_dataset, replacement=True, num_samples=mu * iters * batch_size)
    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=sampler_u,
        batch_size=batch_size*mu,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
        )

    eval_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=True,
        batch_size=5*batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
        )

    sampler_te = RandomSampler(labeled_dataset, replacement=True, num_samples=n_lab_pc)
    train_eval_loader = DataLoader(
        labeled_dataset,
        sampler=sampler_te,
        batch_size=5*batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    return labeled_trainloader, unlabeled_trainloader, eval_loader, train_eval_loader