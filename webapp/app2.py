# -*- coding: utf-8 -*-
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

import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('button.html')

@app.route('/run_script', methods=['POST'])
def run_script():

    # データローダーをインスタンス化
    data_loader_instance = test_dataloader(root=dir_path, batch_size=batch_size, num_class=class_num, random_numbers=random_numbers)

    # run メソッドを呼び出してデータローダーと解答位置を取得
    test_loader, answer_position = data_loader_instance.run()

    print('| Building net')
    net = create_model()

    index, targes, images = test(net, test_loader)
    print(index)
    print(targes)
    print(images)
    print(answer_position)
   
#     results = {
#     '使用した重み': opt.pth_path2,
#     '処理時間（秒）': training_time,
#     'TPR-FPR': TPR - FPR,
#     'TP': int(TP),  # NumPy int64 to Python int
#     'FN': int(FN),
#     'FP': int(FP),
#     'TN': int(TN)
# }

#     if TP != 0:
#         results.update({
#             '精度': float(accuracy_score(y_true, y_pred)),  # Ensure it's a standard float
#             '適合率': float(precision_score(y_true, y_pred)),
#             '再現率': float(recall_score(y_true, y_pred)),
#             'F値': float(f1_score(y_true, y_pred)),
#             '特異度': float(TN / (TN + FP))  # Calculate and convert to float
#         })
    result = {
            '使用した重み': './ResNet10/1/Discriminator-best2.pth',
            '処理時間（秒）': 0.3628864288330078,
            'TPR-FPR': 0.6666666666666667,
            'TP': 2,
            'FN': 0,
            'FP': 1,
            'TN': 2,
            '精度': 0.8,
            '適合率': 0.6666666666666666,
            '再現率': 1.0,
            'F値': 0.8,
            '特異度': 0.6666666666666666
        }
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
