import random
import numpy as np
from PIL import Image
import torch
import os
import shutil


# データセットへのフォルダーへのパス
root_dir = '../../dataset/'
# Imagenetの画像のあるフォルダーへのパス
root_image = root_dir + 'copy/'
# クラス数
# num_class = 20 
target_dic=['ホホジロザメ', 'シュモクザメ', 'ダチョウ', 'カササギ', 'フクロウ', \
       'コウノトリ', 'フラミンゴ', 'トラ猫', '木彫り兎', '牛']
label_dic=[2,4,9,18,24,128,130,283,331,345]
target2_dic=['イタチザメ', 'アカエイ', 'アトリ', 'コガラ', 'ハゲワシ', \
        'ヘラサギ', 'コアオサギ', 'ペルシャ猫', '野ウサギ', '水牛']
label2_dic=[3,5,10,19,23,129,131,284,332,346]
mkdir = os.path.join('.', 'dataset')
count_images=[0]*len(label_dic)
num_images=10
# imageとラベルを一致させるファイルを開く          
os.makedirs(mkdir, exist_ok = True)
with open(os.path.join(root_dir, 'info', 'imagenet_val.txt')) as f:
    lines = f.readlines()
    for line in lines:
        # image名とラベル名を切り取る
        img, target = line.split()
        target = int(target)
        # ラベルが指定したクラス内の写真を選ぶ
        if target in label_dic:
            index = label_dic.index(target)
            # 画像枚数がnum_imagesを超えないようにする
            if count_images[index] < num_images:
                # 無ければそのクラス用のフォルダーを作る
                os.makedirs(os.path.join(mkdir, str(target_dic[index])), exist_ok = True)
                # 画像のコピ-
                shutil.copyfile(os.path.join(root_image, 'val', img), 
                                os.path.join(mkdir, str(target_dic[index]), 
                                        str(count_images[index]) + '.JPEG'))
                count_images[index]+=1
        if target in label2_dic:
            index = label2_dic.index(target)
            # 画像枚数がnum_imagesを超えないようにする
            if count_images[index] < num_images:
                # 無ければそのクラス用のフォルダーを作る
                os.makedirs(os.path.join(mkdir, str(target2_dic[index])), exist_ok = True)
                # 画像のコピ-
                shutil.copyfile(os.path.join(root_image, 'val', img), 
                                os.path.join(mkdir, str(target2_dic[index]), 
                                        str(count_images[index]) + '.JPEG'))
                count_images[index]+=1