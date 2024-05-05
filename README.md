# hackathon

# Webアプリケーション

このリポジトリは、Pythonを使用したWebアプリケーションの基本的な構造を示しています。

## 主要なファイルとディレクトリ
- `main`:プログラムを起動し、webアプリを実行するディレクトリ
    - `app.py`: これが本流のプログラムであり、Webアプリの起動に使います。
    - `data_loader.py`: このスクリプトは、`app.py`で使用されるデータセットを作成します。
    - `static/`: このディレクトリは、Webアプリケーションで使用される画像ファイルを保存します。
        - `dataset`:このディレクトリにはimagenetから抽出したデータセットが格納されています。このデータセットを元にdata_loader.pyによってapp.pyで使用する形に変換されます。
        - `images`:htmlで使用する画像を保存しています。
        -`style.css`:htmlの形式を整えます。
    - `templates/`: このディレクトリには、Webアプリケーションで表示するためのHTMLファイルが保存されます。
    - `button.html`: アプリのスタート画面です。
    - `quiz.html`: クイズを行う画面です。
    - `result.html`: クイズの結果を表示する画面です。
    - `select_model.html`:使用するモデルを選択する画面です。
    
- `DNN`:mainを作るときのtestコードです.

## 使用方法

このWebアプリケーションをローカルで実行する手順は以下の通りです。

1. ターミナルを開き、プロジェクトのメインディレクトリに移動します。

    ```bash
    cd path/to/your/project/main
    ```

2. 以下のコマンドを実行して、Webアプリケーションを起動します。

    ```bash
    python app.py
    ```

3. アプリケーションが起動したら、表示されるURL（通常は `http://localhost:5000`）をクリックするか、Webブラウザに直接入力してアクセスします。

4. ブラウザが開いたら、アプリケーションのスタート画面が表示されます。ここから各機能を利用できます。

# topa'z
ここからは
https://topaz.dev/projects/76445eb059bc90572dc3
と同じ内容です

# 君たちはAIに勝てるか！？
私たちは普段pythonを使って機械学習の研究をしています！
今回は各年代の画像分類モデルの中で幅をきかせていた番長モデルと戦ってもらいます！

# 概要
1. ゲーム開始前に「名前,モデルの難易度,画像枚数」を選択します.*
2. スタートボタンを押し、ゲームスタートです。
3. 選択肢の中から仲間はずれの画像を選択します。
4. 正誤判定とAIの予測結果、さらに正解の画像が表示されます。
（全部で問題は10問。問題の種類は変わりませんが、毎回選択肢が変化します。）
5. 10問の回答が終わると結果が表示されます。AIに勝つことができればランキングに名前と得点が反映されます。
　ランキング上位を目指してより強力なAIに挑みましょう！！！

*1問10点×モデルの難易度(1~4)×画像枚数÷2で得点を計算します。(引き分けの場合スコアは半分)

## 君を待つ強力なAIたち

### AlexNet (2012)  - Easy
AlexNetは深層学習の歴史を塗り替えた革命的なモデル。2012年に、画像認識の大会で圧倒的な成績を収め、AIの可能性に世界中の目を見張らせた。
### VGG16 (2014) - Normal
VGGチームによる16層の深い構造は、シンプルながらも強力。3x3の小さなフィルタを重ねることで、精度の高い特徴抽出を実現した。
### ResNet50 (2015) - Hard
深い層が生む勾配消失の問題を、残差回路で上手く解決。50層の深さにも関わらず、情報が滑らかに伝播し、画期的な性能を発揮する。
### Efficient-Net (2019) - Very Hard
モバイル環境でも高い認識精度を維持できるよう設計された革新的なアーキテクチャ。計算効率を損なうことなく、主要タスクで最先端の性能を実現する。

## クイズに使用した画像
大規模データセットであるImageNetのテストデータに含まれる画像から新たに作成したデータセットを使用している。データセットはフォルダー名がクラス名となっており、その中に10枚ずつ画像が入っている。クイズに使用する画像は作成したデータセットからクラスごとにランダムに取得している。

## 仲間外れ画像探索アルゴリズム
AIが仲間外れの画像を特定する手順は以下のようになっている。
1. ImageNetで事前学習済みを行ったモデルをダウンロードする。
2. それぞれの画像に対して、モデルの予測確率を出力する。この予測確率は入力された画像に対して、モデルがそのクラスに属すると予測した確率になっている。
3. 1つの問題に出題するすべての画像に対する予測確率の和をクラスごとに計算し、最大となったクラスを多数派の画像の予測クラスとする。
4. 入力画像の内、多数派の画像の予測クラスに対する予測確率が最も低い画像を仲間外れの画像とする。

## 実装環境
- osはwindowsを使用しました。
- 画像分類タスク、本流の流れはpythonを使用しました。
- Flask、HTML、JS、CSSを用いてwebアプリとして実装しました。
- Github、VSCodeを用いて情報の共有、共同開発を行いました。

## 苦労した点
- なかなかアイデアが出なかった。
- フロントエンドの開発経験が無かったため、HTMLのコーディングに苦労した。
- 共同開発経験に乏しく、GitHubの使い方に戸惑った。

## 技術の無駄遣い
- 画像枚数に応じたスマートな画像配置を実現！
- 正解画像から後光が差し込む…！
- 画面に彩りを与える愉快なSEやイラストを豊富に用意！