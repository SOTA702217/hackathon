from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score  # sklearnから様々な評価指標をインポート
from sklearn.metrics import roc_curve, roc_auc_score  # ROC曲線とAUCスコアの計算用関数をインポート
import matplotlib.pyplot as plt  # グラフ描画のためのmatplotlib.pyplotをインポート

import numpy as np  # 数値計算のためのnumpyをインポート
import torch  # PyTorchの基本モジュールをインポート

from torchvision import transforms, models  # torchvisionからtransformsとmodelsをインポート

import os  # ファイルパス操作のためのosモジュールをインポート
from flask import Flask, request, render_template, redirect, url_for, jsonify  # Flask関連の機能をインポート
from PIL import Image  # 画像操作のためのPillowライブラリをインポート
from werkzeug.utils import secure_filename  # ファイル名を安全に扱うための関数をインポート

from data_loader import test_dataloader  # データローディング用のカスタム関数をインポート
import random  # 乱数生成のためのrandomモジュールをインポート
import csv  # CSVファイル操作のためのcsvモジュールをインポート

from flask_sqlalchemy import SQLAlchemy  # FlaskとSQLAlchemyの統合をサポートする拡張モジュールをインポート

app = Flask(__name__)  # Flaskアプリケーションのインスタンスを作成
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///rankings.db'  # SQLiteデータベースのURIを設定
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # SQLAlchemyの変更追跡機能を無効に設定
db = SQLAlchemy(app)  # SQLAlchemyのインスタンスを作成
# データセットへのパス
dir_path = './static/dataset'
# 一度に表示する画像の枚数
# batch_size = 4
# 各クラスの写真枚数
num_sample = 10
# クラス数(問題数に対応)
class_num = 10
#modelの重み
difficulty=1

class Ranking(db.Model):  # データベースのランキングテーブルのモデルクラス
    id = db.Column(db.Integer, primary_key=True)  # ユニークなID
    player_name = db.Column(db.String(80), nullable=False)  # プレイヤー名
    score = db.Column(db.Integer, nullable=False)  # スコア

    def __repr__(self):  # オブジェクトの文字列表現を定義
        return f"<Ranking {self.player_name} score {self.score}>"

with app.app_context():  # アプリケーションコンテキスト内で
    db.create_all()  # データベーステーブルを作成

# モデル名とその設定を辞書で対応付け
model_type = {
    'alexnet': models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1'),  # AlexNetモデル
    'vgg16': models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1'),  # VGG16モデル
    'resnet50': models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2'),  # ResNet50モデル
    'efficientnet-b7': models.efficientnet_b7(weights='EfficientNet_B7_Weights.IMAGENET1K_V1')  # EfficientNet-B7モデル
}

def add_ranking(player_name, score):  # ランキングに新しいエントリを追加する関数
    new_ranking = Ranking(player_name=player_name, score=score)
    db.session.add(new_ranking)
    db.session.commit()

def get_rankings():  # ランキングを取得する関数
    rankings = Ranking.query.order_by(Ranking.score.desc()).all()
    return rankings

def test(net, loader):  # ネットワークでテストデータを評価する関数
    net.eval()
    min_index_list = []
    targes_list = []
    images_list = []
    with torch.no_grad():  # 勾配計算を無効化
        for _, (img_path, inputs, targets, _) in enumerate(loader):
            outputs = net(inputs)
            out_soft = torch.softmax(outputs, dim=1)
            sum_out = torch.sum(out_soft, dim=0)
            _, predicted = torch.max(sum_out, dim=0)
            _, min_index = torch.min(out_soft, dim=0)
            min_index_list.append(int(min_index[predicted]))
            targes_list.append(targets)
            images_list.append(img_path)
    return min_index_list, targes_list, images_list

def create_model(models_name):  # モデル名からモデルを作成する関数
    model = model_type.get(models_name)
    return model 

# 以下の@app.routeデコレータを使用した関数は、Flaskのルート（URLエンドポイント）を定義し、特定のURLにアクセスがあった際に実行される関数です。
@app.route('/')  # ホームページのルートを定義
def index():
    return render_template('button.html')  # button.htmlをレンダリングして表示

@app.route('/select_model', methods=['POST'])  # モデル選択ページのルートを定義
def select_model():
    return render_template('select_model.html')  # select_model.htmlをレンダリングして表示

@app.route('/quiz', methods=['POST'])  # クイズページのルートを定義
def quiz():
    global difficulty
    player_name = request.form['name']  # フォームからプレイヤー名を取得
    selected_model = request.form['model']  # 選択されたモデル名を取得
    if selected_model == 'alexnet':  # 選択されたモデルに応じて難易度を設定
        difficulty = 1
    elif selected_model == 'vgg16':
        difficulty = 2
    elif selected_model == 'resnet50':
        difficulty = 3
    else:
        difficulty = 4

    image_count = int(request.form['image_count'])  # 画像の枚数をフォームから取得
    random_numbers = random.sample(range(num_sample), image_count)  # ランダムに画像を選択
    data_loader_instance = test_dataloader(root=dir_path, batch_size=image_count, num_class=class_num, random_numbers=random_numbers)  # データローダーインスタンスを作成
    test_loader, answer_pos = data_loader_instance.run()  # データローダーを実行して、テストローダーと解答位置を取得

    net = create_model(selected_model)  # 選択されたモデルでネットワークを作成

    predict_pos, label, image_paths_list = test(net, test_loader)  # テスト関数を実行して、AIの予測位置、ラベル、画像パスを取得
    nakamahazure_labels = []
    nakama_labels = []
    for index, i in enumerate(answer_pos):
        nakamahazure_labels.append(label[index][i])
        nakama_labels.append(label[index][i - 1])

    quizzes = []  # クイズデータを格納するリスト
    for paths, ai_pred, correct_ans, nakamahazure, nakama in zip(image_paths_list, predict_pos, answer_pos, nakamahazure_labels, nakama_labels):
        image_paths = ['dataset/' + path for path in paths]  # 画像パスを整形
        quiz = {
            'image_paths': image_paths,
            'ai_answer': ai_pred,
            'correct_answer': correct_ans,
            'nakamahazure': nakamahazure,
            'nakama': nakama
        }
        quizzes.append(quiz)  # クイズリストに追加

    return render_template('quiz.html', quizzes=quizzes, batch_size=image_count, player_name=player_name, model=select_model)  # quiz.htmlをレンダリングして表示

@app.route('/results', methods=['GET', 'POST'])  # 結果ページのルートを定義
def results():
    global difficulty
    player_name = request.args.get('player_name', default="", type=str)  # URLパラメータからプレイヤー名を取得
    player_score = request.args.get('playerScore', default=0, type=int)  # URLパラメータからプレイヤースコアを取得
    ai_score = request.args.get('aiScore', default=0, type=int)  # URLパラメータからAIスコアを取得
    batch_size = request.args.get('batchsize', default=0, type=int)  # URLパラメータからバッチサイズを取得
    ai_score = (ai_score * 5) * (batch_size) * (difficulty)  # AIスコアを計算
    player_score = (player_score * 5) * (batch_size) * (difficulty)  # プレイヤースコアを計算
    if player_score > ai_score:
        add_ranking(player_name, player_score)  # プレイヤースコアがAIスコアより高い場合、ランキングに追加
    elif player_score == ai_score:
        add_ranking(player_name, player_score / 2)  # スコアが同じ場合、半分のスコアでランキングに追加

    return render_template('result.html', player_name=player_name, player_score=player_score, ai_score=ai_score)  # result.htmlをレンダリングして表示

@app.route('/rankings')  # ランキングページのルートを定義
def show_rankings():
    player_name = request.args.get('player_name', default="", type=str)  # URLパラメータからプレイヤー名を取得
    player_score = request.args.get('playerScore', default=0, type=int)  # URLパラメータからプレイヤースコアを取得
    ai_score = request.args.get('aiScore', default=0, type=int)  # URLパラメータからAIスコアを取得
    rankings = get_rankings()  # データベースからランキングを取得
    return render_template('rankings.html', player_name=player_name, player_score=player_score, ai_score=ai_score, rankings=rankings)  # rankings.htmlをレンダリングして表示

if __name__ == '__main__':
    app.run(debug=True)  # アプリケーションをデバッグモードで実行
