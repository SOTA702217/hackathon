# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import random
import os
import torchvision
import torchvision.models as models
from dataloader import test_dataloader

model_name= 'AlexNet'
dir_path = './dataset'
# 一回で表示する画像数に対応
batch_size = 4
num_sample = 10
class_num = 10
random_numbers=random.sample(range(10), 4)

# モデル名と種類の辞書
model_type = {
    'resnet50' : models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2'), # top1:80.9
    'efficientnet-b7': models.efficientnet_b7(weights='EfficientNet_B7_Weights.IMAGENET1K_V1'), # top1:69.8
    'VGG16': models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1'), # top1:71.6
    'alexnet': models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1') # top1:56.5
}

# ネットワークの予測を出力
def test(net, loader):
    net.eval()
    min_index_list=[]
    targes_list=[]
    images_list=[]
    with torch.no_grad():
        for _, (img_path, inputs, targets, _) in enumerate(loader):
            # DNNの予測
            outputs = net(inputs)
            # 予測を確率に
            out_soft = torch.softmax(outputs, dim=1)
            # クラスごとにバッチ内の予測確率の和をとる
            sum_out = torch.sum(out_soft, dim=0)
            # バッチ内で最も予測確率が高いクラスを特定
            _, predicted = torch.max(sum_out, dim=0)
            # それぞれのクラスごとにもっと低い予測をしたサンプルのインデックスを取得
            _, min_index = torch.min(out_soft, dim=0)
            # バッチ内で最も予測確率が高いクラスの中で，
            # 最も予測確率が低いサンプルのインデックスを取得し，リストとして保存
            min_index_list.append(int(min_index[predicted]))
            # 正解ラベルをリストとして保存
            targes_list.append(targets)
            # 写真へのパスを保存
            images_list.append(img_path)

    return min_index_list, targes_list, images_list

# モデルのダウンロード
def create_model(models_name):
    model = model_type.get(models_name)
    return model 

# データローダーをインスタンス化
loader = test_dataloader(root=dir_path,batch_size=batch_size,  num_class=class_num, random_numbers=random_numbers)
# データローダー作成
test_loader, answer_position  = loader.run()

# モデルをビルド
print('| Building net')
net = create_model(model_name)
# ネットワークの予測
index, targes, images = test(net, test_loader)
print(index)
print(targes)
print(images)