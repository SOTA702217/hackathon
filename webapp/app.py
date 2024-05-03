# -*- coding: utf-8 -*-
import torch
import random
import os
import torchvision
import torchvision.models as models
from dataloader import test_dataloader

model_name= 'AlexNet'
dir_path = './dataset'
batch_size = 4
num_sample = 10
class_num = 10
random_numbers=random.sample(range(10), 4)

# モデル名と種類の種類の辞書
model_type = {
    'ResNet50' : models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2'),
    'ResNet18': models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1'),
    'VGG16': models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1'),
    'AlexNet': models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
}

import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('button.html')

@app.route('/run_script', methods=['POST'])
def run_script():
     # ユーザーが選択したモデル名を取得
    model_name = request.form.get('model_name')

   # データローダーをインスタンス化
    loader = test_dataloader(root=dir_path,batch_size=batch_size,  num_class=class_num, random_numbers=random_numbers)
    # データローダー作成
    test_loader, answer_position  = loader.run()

    # print('| Building net')
    net = create_model(model_name)

    index, targes, images = test(net, test_loader)
    # print(index)
    # print(targes)
    # print(images)
    # print(answer_position)

    count=sum(a==b for a,b in  zip(index,answer_position))
    par=int(count*10)
   
    result = {
            '予測画像': index,
            'ラベル': targes,
            'パス': images,
            '正解バッチ': answer_position,
            '正解率': str(par)+'%'
        }
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
