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

import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

from data_loader import test_dataloader
import random
import csv
import os

# CSVファイルのパス
file_path = 'rankings.csv'

# CSVファイルを作成して初期データを書き込む
# with open(file_path, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     # 例として、ヘッダーを書き込む（必要に応じて）
#     writer.writerow(['Player Name', 'Score'])
#     # 現在の作業ディレクトリを確認

if not os.path.exists(file_path):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 例として、ヘッダーを書き込む（必要に応じて）
        writer.writerow(['Player Name', 'Score'])
print(os.getcwd())

# ファイルへのパスが正しいか確認
print(os.path.exists(file_path))

try:
    with open(file_path, mode='r') as file:
        # ファイルの読み込み処理
        pass
except FileNotFoundError:
    print(f"Error: The file {file_path} does not exist.")

def print_csv_content(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)

print_csv_content('rankings.csv')


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

def load_rankings(file_path):
    rankings = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader, None)  # ヘッダ行を読み飛ばす
        for row in reader:
            if row and len(row) == 2:  # 行が空でなく、要素が2つあることを確認
                try:
                    rankings.append([row[0], int(row[1])])
                except ValueError:
                    print(f"Warning: Skipping invalid score data in row: {row}")
    print("Loaded rankings:", rankings)  # デバッグ情報
    return rankings

def csv_to_list(file_path):
    data_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # ヘッダー行をスキップ
        for row in csv_reader:
            name = row[0].strip()  # 余白を削除
            score = int(row[1])  # スコアを整数に変換
            data_list.append([name, score])

    return data_list

rankings = csv_to_list('rankings.csv')


# 初期ランキングのロード
# rankings = load_rankings('rankings.csv')
# print(rankings)

def save_rankings(rankings, file_path):
    try:
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Player Name', 'Score'])
            for ranking in rankings:
                writer.writerow(ranking)
        print("Rankings saved successfully.")
    except Exception as e:
        print(f"Error saving rankings: {e}")


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
    player_name = request.form['name']
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
    # print(player_name)
    return render_template('quiz.html',player_name=player_name, quizzes=quizzes)

@app.route('/results', methods=['GET', 'POST'])
def results():
    global rankings
       
    player_name = request.args.get('player_name', default="", type=str)
    player_score = request.args.get('playerScore', default=0, type=int)
    
    # スコアをランキングに追加
    rankings.append([player_name, player_score])
    # スコアでソート（降順）
    print(rankings)
    rankings.sort(key=lambda x: int(x[1]), reverse=True)
    # トップ10のみを保存
    if len(rankings) > 10:
        rankings = rankings[:10]
    
    # ランキングをCSVファイルに保存
    save_rankings(rankings, 'rankings.csv')
    
    player_name = request.args.get('player_name', default="", type=str)
    player_score = request.args.get('playerScore', default=0, type=int)
    ai_score = request.args.get('aiScore', default=0, type=int)
    
    return render_template('result.html', player_name=player_name, player_score=player_score, ai_score=ai_score)

@app.route('/rankings')
def show_rankings():
    rankings = load_rankings('rankings.csv')
    print(rankings)
    if not rankings:
        print("No rankings available.")
    return render_template('rankings.html', rankings=rankings)


if __name__ == '__main__':
    app.run(debug=True)