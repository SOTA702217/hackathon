import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# モデルの選択肢
model_choices = {
    'alexnet': (models.alexnet, models.AlexNet_Weights.DEFAULT),
    'vgg16': (models.vgg16, models.VGG16_Weights.DEFAULT),
    'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT),
    'inception_v3': (models.inception_v3, models.Inception_V3_Weights.DEFAULT),
    'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT)
}

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description='Image classification using pre-trained models')
parser.add_argument('--model', choices=model_choices.keys(), default='resnet50',
                    help='Pre-trained model to use (default: resnet50)')
parser.add_argument('--image_path', type=str, required=True,
                    help='Path to the input image')
args = parser.parse_args()

# 選択されたモデルをロード
model_fn, weights = model_choices[args.model]
model = model_fn(weights=weights)
model.eval()

# 入力画像の前処理
def preprocess_image(image_path, model_name):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if model_name == 'inception_v3':
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

# ImageNetのラベルファイルを読み込む

labels_file = 'for_bunruitest/imagenet_classes.txt'
with open(labels_file, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 画像の前処理
input_image = preprocess_image(args.image_path, args.model)

# モデルによる予測
with torch.no_grad():
    output = model(input_image)
    _, predicted = torch.max(output, 1)

# 予測結果の表示
predicted_label = labels[predicted.item()]
print(f'Predicted label: {predicted_label}')
print(f'Model used: {args.model}')