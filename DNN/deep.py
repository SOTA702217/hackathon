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

dir_path = './dataset'
batch_size = 4
num_sample = 10
class_num = 10
random_numbers=random.sample(range(10), 4)


# ネットワークの予測を出力
def test(net, loader):
    net.eval()
    min_index_list=[]
    targes_list=[]
    images_list=[]
    with torch.no_grad():
        for batch_idx, (img_path, inputs, targets, labels) in enumerate(loader):
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
            # 
            targes_list.append(targets)
            images_list.append(img_path)

    return min_index_list, targes_list, images_list

# モデルをビルド
def create_model():
    model = models.resnet50(pretrained=True)
    return model 

# model_name = {
#     'ResNet50': [models.resnet50(pretrained=True)],
#     'VGG16': [models.alexnet('resnet18', cifar=True)],
#     'bigresnet50': [resnet.ResNet('resnet18', cifar=True), 2048],
#     'bigresnet18_preact': [preact_resnet.ResNet18, 512],
#     'resnet18': [resnet.ResNet('resnet18'), 512],
#     'resnet34': [resnet.ResNet('resnet34'), 512],
#     'resnet50': [resnet.ResNet('resnet50'), 2048],
#     # 追加
#     'Coresnet50': [resnet.ResNet('resnet50', preact=True), 2048],
# }

# データローダーをインスタンス化
loader = test_dataloader(root=dir_path,batch_size=batch_size,  num_class=class_num, random_numbers=random_numbers)
test_loader = loader.run()

print('| Building net')
net = create_model()

index, targes, images = test(net, test_loader)
print(index)
print(targes)
print(images)