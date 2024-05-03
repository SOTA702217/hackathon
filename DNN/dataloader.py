#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:59:00 2024

@author: reo
"""
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class test_dataset(Dataset):
    def __init__(self, root, transform, num_samples, num_class):    
       self.root = root
       self.transform = transform
       self.target = []
       self.label = []
       self.images = []  
       target_dic=['うさぎ', '猫', '牛', 'ホホジロサメ', 'コウノトリ', \
       'フラミンゴ', 'シュモクザメ', 'フクロウ', 'カササギ', 'ダチョウ']
       label_dic=[331, 283, 345, 2, 128, 130, 4, 24, 18, 9]
       
       for i in range(0, num_class):
           for k in range(0, num_samples):
               self.target.append(target_dic[i])
               self.label.append(label_dic[i])
               self.images.append(os.path.join(target_dic[i], str(k)+'.JPEG'))
       
    def __getitem__(self, index):  
       print(self.images)
       exit()
       img_path = self.images[index]
       targets = self.target[index]     
       labels = self.label[index]
       print(img_path)
       image = Image.open(os.path.join(self.root,img_path)).convert('RGB') 
       img = self.transform(image) 
       return img, targets, labels
    
    def __len__(self):
        return len(self.images)       
    

class test_dataloader():  
    def __init__(self, root, batch_size, num_samples, num_class):    
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_class = num_class
        self.root = root
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        
    def run(self,pred=[],prob=[],paths=[]):        
        test_datasets = test_dataset(self.root, transform=self.transform,
                                     num_samples=self.num_samples, num_class=self.num_class)
        test_loader = DataLoader(
            dataset=test_datasets, 
            batch_size=self.batch_size,
            shuffle=False,)             
        return test_loader             
