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


# ネットワークの予測を出力
def test(net, loader):
    net.eval()
    predict = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, labels) in enumerate(loader):
            # DNNの予測
            outputs = net(inputs)
            # 予測を確率に
            out_soft = torch.softmax(outputs, dim=1)
            # クラスごとにバッチ内の予測確率の和をとる
            sum_out = torch.sum(out_soft, dim=0)
            # バッチ内で最も予測確率が高いクラスを特定
            _, predicted = torch.max(sum_out, dim=0)
            # _, predicted = torch.max(outputs, 1)            
            # predict.append(predicted)
            # print(batch_idx)

    return predict

# モデルをビルド
def create_model():
    model = models.resnet50(pretrained=True)
    return model 


loader = test_dataloader(root=dir_path,batch_size=batch_size, num_samples=num_sample, num_class=class_num)
test_loader = loader.run()

print('| Building net')
net = create_model()

pre = test(net, test_loader)
print(pre)