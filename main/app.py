import argparse
# import glob
# import os

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# import cv2
import numpy as np
import torch

from torchvision import transforms
from torchvision import models
# import glob

import time
# import psutil

import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

from data_loader import test_dataloader
import random

# static_folderを指定しているのは、画像ファイルを表示するため
# app = Flask(__name__, static_folder = "./static/")
app = Flask(__name__)

dir_path = './static/dataset'
batch_size = 4
num_sample = 10
class_num = 10
random_numbers=random.sample(range(10), 4)


model_type = {
    'alexnet': models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1'), # top1:56.5
    'vgg16': models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1'), # top1:69.8  
    'resnet50' : models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2'), # top1:80.9
    'efficientnet-b7': models.efficientnet_b7(weights='EfficientNet_B7_Weights.IMAGENET1K_V1') # top1:84.1
}


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
# def create_model():
#     model = models.resnet50(pretrained=True)
#     return model 

def create_model(models_name):
    model = model_type.get(models_name)
    return model 

@app.route('/')
def index():
    return render_template('button.html')

@app.route('/select_model', methods=['POST'])
def select_model():
    return render_template('select_model.html')

@app.route('/quiz', methods=['POST'])
def quiz():
    selected_model = request.form['model']
        # データローダーをインスタンス化
    data_loader_instance = test_dataloader(root=dir_path, batch_size=batch_size, num_class=class_num, random_numbers=random_numbers)

    # run メソッドを呼び出してデータローダーと解答位置を取得
    test_loader, answer_pos = data_loader_instance.run()

    print('| Building net')
    net = create_model(selected_model)

    # predict_pos : AIの予測位置
    # label : ラベル
    # image_paths_list :画像のパス
    # answer_pos : 実際の正解位置
    predict_pos, label, image_paths_list = test(net, test_loader)
    nakamahazure_labels = []
    nakama_labels = []
    for index, i in enumerate(answer_pos):
        nakamahazure_labels.append(label[index][i]) 
        nakama_labels.append(label[index][i-1])

    quizzes = []
    for paths, ai_pred, correct_ans,nakamahazure, nakama in zip(image_paths_list, predict_pos, answer_pos, nakamahazure_labels, nakama_labels):
        image_paths = ['dataset/' + path for path in paths]
        quiz = {
            'image_paths': image_paths,
            'ai_answer': ai_pred,
            'correct_answer': correct_ans,
            'nakamahazure': nakamahazure,
            'nakama': nakama
        }
        quizzes.append(quiz)

    print(quizzes)
    print(len(quizzes))
    return render_template('quiz.html', quizzes=quizzes)

@app.route('/results', methods=['GET', 'POST'])
def results():
    player_score = request.args.get('playerScore', 0, type=int)
    ai_score = request.args.get('aiScore', 0, type=int)
    
    return render_template('result.html', player_score=player_score, ai_score=ai_score)

if __name__ == '__main__':
    app.run(debug=True)
