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
    def __init__(self, root, transform, num_class,random_numbers):    
        self.root = root
        self.transform = transform
        self.target = []
        self.label = []
        self.images = [] 
        self.random_numbers =random_numbers 
        self.answer_position=[]
        # ラベル名とラベルのインデックス番号のリストを用意
        target_dic=['ホホジロザメ', 'シュモクザメ', 'ダチョウ', 'カササギ', 'フクロウ', \
                    'コウノトリ', 'フラミンゴ', 'トラ猫', '木彫り兎', '牛']
        label_dic=[2,4,9,18,24,128,130,283,331,345]
        # 間違え側のサンプルのラベル名とラベルインデックス番号
        target_dic_near=['イタチザメ', 'アカエイ', 'アトリ', 'コガラ', 'ハゲワシ', \
                    'ヘラサギ', 'コアオサギ', 'ペルシャ猫', '野ウサギ', '水牛']
        label_dic_near=[3,5,10,19,23,129,131,284,332,346]
        # 3つの仲間サンプルと1つの仲間はずれサンプルの組みを作成
        # それぞれのクラスごろにループを回す
        for i in range(0, num_class):
            j=0
            # 仲間外れのサンプルが何番目に来るかを乱数で決めている
            l=random.randint(0, 3)
            # 正解の場所がどこかを保存する
            self.answer_position.append(l)
            # random_numdersに入っている写真を取得
            for k in self.random_numbers:
                # 仲間外れのサンプルを取得
                if j!=l:
                    self.target.append(target_dic[i])
                    self.label.append(label_dic[i])
                    self.images.append(os.path.join(target_dic[i], str(k)+'.JPEG'))
                # 仲間のいるサンプルを取得
                else:
                    self.target.append(target_dic_near[i])
                    self.label.append(label_dic_near[i])
                    self.images.append(os.path.join(target_dic_near[i], str(k)+'.JPEG'))
                j+=1
    # バッチごとにサンプルを追加
    def __getitem__(self, index):  
       img_path = self.images[index]
       targets = self.target[index]     
       labels = self.label[index]
       image = Image.open(os.path.join(self.root,img_path)).convert('RGB') 
       img = self.transform(image) 
       # returnはそれぞれ，画像へのパス，画像，正解ラベル，正解インデックス
       return img_path, img, targets, labels
    
    def __len__(self):
        return len(self.images)       
    
# データローダー
class test_dataloader():  
    def __init__(self, root, batch_size, num_class,random_numbers):    
        # 1度に表示する画像
        self.batch_size = batch_size
        # クラス数
        self.num_class = num_class
        # データセットへのパス
        self.root = root
        # 選択する写真を決める
        self.random_numbers =random_numbers 
        # 画像サイズの変更，DNNに入力できるように画像の形を変更
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    
    # データセットの作成
    def run(self):        
        test_datasets = test_dataset(self.root, transform=self.transform,
                                     num_class=self.num_class,random_numbers=self.random_numbers)
        test_loader = DataLoader(
            dataset=test_datasets, 
            batch_size=self.batch_size,
            shuffle=False,)             
        return test_loader, test_datasets.answer_position             