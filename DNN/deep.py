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

dir_path = os.path.join('.', 'dataset')
batch_size = 4
random_numbers=random.sample(range(10), 4)#randamな4要素の配列



# ネットワークの予測を出力
def test(net, loader):
    net.eval()
    predict = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (img_path,inputs, targets, labels) in enumerate(loader):
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)            
            predict.append(predicted)
            print(batch_idx)

    return predict

# モデルをビルド
def create_model():
    # ここに名前入れるように
    model = models.resnet50(pretrained=True)
    return model 


loader = test_dataloader(root=dir_path,batch_size=batch_size, num_class=10,random_numbers=random_numbers)
test_loader,answer_position = loader.run()
# print(answer_position)
# print(random_numbers)
print('| Building net')
net = create_model()
pre = test(net, test_loader)
print(pre)