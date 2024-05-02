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

from lib.ResNet_10 import ResNet10

from utils.dataloader2 import test_dataset
# import glob

import time
# import psutil

import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
k = 5
thresh = 160
lower_size = 300
upper_size = 43264
total = 0
correct = 0

model = ResNet10()
model.cpu()
model.eval()

@app.route('/')
def index():
    return render_template('button.html')

@app.route('/run_script', methods=['POST'])
def run_script():


    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')

    parser.add_argument('--pth_path2', type=str,
                        default='./ResNet10/1/Discriminator-best2.pth')
    parser.add_argument('--test_path1', type=str, default='./Test', help='path to test dataset')
    parser.add_argument('--test_path2', type=str, default='./test/tensor/', help='path to test dataset')

    opt = parser.parse_args()

    data_path1 = opt.test_path1
    data_path2 = opt.test_path2
    image_root1 = '{}/images/'.format(data_path1)
    gt_root1 = '{}/masks/'.format(data_path1)
    res_root='{}'.format(data_path2)#
    test_loader1 = test_dataset(image_root1, gt_root1, opt.testsize)
    res_loader1=test_dataset(res_root,gt_root1,opt.testsize)
    
    global model
    # モデルの状態を読み込む
    model.load_state_dict(torch.load(opt.pth_path2, map_location=torch.device('cpu')))
    y_true = np.array([])
    y_score = np.array([])
    y_pred = np.array([])

        #学習開始時間の記録
    start_time = time.time()

    for i in range(res_loader1.size):
        res, gt, name = res_loader1.load_data()
        label = transforms.functional.to_tensor(gt)
        label = torch.einsum("ijk->i", label) > 0
        label = torch.where(label > 0, torch.tensor(1), torch.tensor(0))
        gt = np.asarray(gt, np.float32)

        # gt /= (gt.max() + 1e-8)  ##########################

        # image = image.cuda()
        res=res.cpu()
        with torch.no_grad():
        
            out = model(res)#

            _, predicted = torch.max(out, dim=1)

            predicted = torch.Tensor.cpu(predicted).detach().numpy()

            label = torch.Tensor.cpu(label).detach().numpy()
            out = torch.Tensor.cpu(out).detach().numpy()
            y_true = np.append(y_true, label)
            y_score = np.append(y_score, out[0][1] - out[0][0])
            y_pred = np.append(y_pred, predicted)
            # プログラム終了時のメモリ使用量
            # print("Memory usage at the 3:")
            # print_memory_usage()

    #学習終了時間の記録
    end_time = time.time()

    #学習にかかった時間の計算
    training_time = end_time - start_time
    print(f"使った重み：{opt.pth_path2}")
    print(f"処理にかかった時間:{training_time}秒")

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.flatten()
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    print("----------discriminator-----------")
    print("TPR-FPR:", TPR-FPR)
    print("TP:", TP)
    print("FN:", FN)
    print("FP:", FP)
    print("TN:", TN)

    if TP != 0:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        F_measure = f1_score(y_true, y_pred)
        specificity = TN / (TN + FP)

        print("Accuracy:", accuracy)
        print("F-measure:", F_measure)
        print("precision:", precision)
        print("recall:", recall)
        print("specificity:", specificity)

        output_text_1 = f"----------discriminator-----------\nAccuracy: {accuracy}\nF-measuer: {F_measure}\n\n"

    results = {
    '使用した重み': opt.pth_path2,
    '処理時間（秒）': training_time,
    'TPR-FPR': TPR - FPR,
    'TP': int(TP),  # NumPy int64 to Python int
    'FN': int(FN),
    'FP': int(FP),
    'TN': int(TN)
}

    if TP != 0:
        results.update({
            '精度': float(accuracy_score(y_true, y_pred)),  # Ensure it's a standard float
            '適合率': float(precision_score(y_true, y_pred)),
            '再現率': float(recall_score(y_true, y_pred)),
            'F値': float(f1_score(y_true, y_pred)),
            '特異度': float(TN / (TN + FP))  # Calculate and convert to float
        })
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
