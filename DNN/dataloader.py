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
import random


class test_dataset(Dataset):
    def __init__(self, root, transform, num_class):    
       self.root = root
       self.transform = transform
       self.target = []
       self.label = []
       self.images = [] 
       random_numbers = random.sample(range(10), 4)#randamな4要素の配列
    #    print(random_numbers) 
       target_dic=['ホホジロザメ', 'シュモクザメ', 'ダチョウ', 'カササギ', 'フクロウ', \
       'コウノトリ', 'フラミンゴ', 'トラ猫', '木彫り兎', '牛']
       label_dic=[2,4,9,18,24,128,130,283,331,345]
       target2_dic=['イタチザメ', 'アカエイ', 'アトリ', 'コガラ', 'ハゲワシ', \
       'ヘラサギ', 'コアオサギ', 'ペルシャ猫', '野ウサギ', '水牛']
       label2_dic=[3,5,10,19,23,129,131,284,332,346]
       for i in range(0, num_class):
           j=0
           for k in random_numbers:
                j+=1
                if j!=4:
                    self.target.append(target_dic[i])
                    self.label.append(label_dic[i])
                    self.images.append(os.path.join(target_dic[i], str(k)+'.JPEG'))
                else:
                    self.target.append(target2_dic[i])
                    self.label.append(label2_dic[i])
                    self.images.append(os.path.join(target2_dic[i], str(k)+'.JPEG'))

    def __getitem__(self, index):  
    #    print(self.images)
       img_path = self.images[index]
       targets = self.target[index]     
       labels = self.label[index]
    #    print(img_path)
       image = Image.open(os.path.join(self.root,img_path)).convert('RGB') 
       img = self.transform(image) 
       return img, targets, labels
    
    def __len__(self):
        return len(self.images)       
    

class test_dataloader():  
    def __init__(self, root, batch_size, num_class):    
        self.batch_size = batch_size
        self.num_class = num_class
        self.root = root
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        
    def run(self,pred=[],prob=[],paths=[]):        
        test_datasets = test_dataset(self.root, transform=self.transform,
                                     num_class=self.num_class)
        test_loader = DataLoader(
            dataset=test_datasets, 
            batch_size=self.batch_size,
            shuffle=False,)             
        return test_loader             
