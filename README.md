# Basketball Game Reconstraction

## プロジェクト概要
### 目的
- ピッチ上のアスリートの位置と識別情報(役割、チーム、ユニフォーム番号)を抽出
- 試合の状況を2Dミニマップ上に再現

### 手法概要
- 選手の画像上の座標を獲得
    - セグメンテーションすることで正確な位置座標の取得
- ピッチ座標に変換（ここではやらない）

### 具体的な手法
1. YOLOv10をファインチューニング
1. SAM2をファインチューニング
1. ファインチューニング済みのYOLOv10を用いて，最初のフレームからbboxを獲得
1. ファインチューニング済みのSAM2に最初のフレームのbboxをプロンプトとして付与
1. SAM2を用いてセグメンテーション
1. 得られたマスクから画像上の位置を推定（まだここまでできていない）

## プロジェクト構造
```
basketball/
├── basketball-video-dataset/             # ここに動画を格納
├── player_detection/
│   ├── fine_tuned_yolo/                  # ファインチューニング済みのモデルを格納
│   ├── predict_frame/                    # 予測した画像が格納
│   └── tools_detection/
│       ├── change_file_name.py
│       ├── check_annotation.py
│       ├── convert_bbox_label.py
│       ├── fine_tuning_yolo.py           # **main file**（ファインチューニング）
│       ├── inference_yolo.py             # **main file**（モデルの推論）
│       └── make_bbox.py
├── player_segmentation/
│   ├── sam2/ ...                         # sam2を動かすのに必要なファイル
│   ├── fine_tuned_sam2/                  # ファインチューニング済みのモデルを格納
│   ├── predict_frame/                    # 予測した画像が格納
│   ├── predict_video/                    # 予測した画像が格納
│   └── tools_segmentation/
│       ├── evaluate_sam2.py
│       ├── fine_tuning_sam.py
│       ├── fine_tuning_sam2.py           # **main file**（ファインチューニング）
│       ├── gen_annotations.py
│       ├── gen_errorbar_graph.py
│       ├── inference_sam.py
│       ├── inference_sam2_with_video.py  # **main file**（モデルの推論）
│       ├── inference_sam2.py             # **main file**（モデルの推論）
│       └── make_mask.py
├── player_tracking/
│   ├── sam2/ ...                         # sam2を動かすのに必要なファイル
│   ├── predict_frame_sam2/               # 予測した画像が格納
│   ├── predict_frame_yolo/               # 予測した画像が格納
│   ├── predict_video/                    # 予測した画像が格納
│   └── tools_tracking/
│       ├── make_video.py
│       ├── modify_tracking_id.py
│       ├── tracking_player_sam2.py       # **main file**（動画に対してマスクを与え続ける）
│       └── tracking_player_yolo.py       # **main file**（動画に対してbboxを与え続ける）
├── .gitignore
└── README.md
```

## player segmentation
### YOLOv10のファインチューニング
fine_tuning_yolo.py を実行．引数は（ごめんなさい．このファイルだけないです．git commitをいじってたら消えました．．）
inference_yolo.py を実行．引数は num_image（画像枚数）, num_time（何回目の学習）

## player detection
### SAM2のファインチューニング
fine_tuning_sam2.py を実行．引数は num_image（画像枚数）, num_epoch（エポック数）
inference_sam2.py を実行．引数は num_image（画像枚数）, --num_train_step（default=best），num_time（何回目の学習）

## player tracking
### ファインチューニング済みのYOLOv10を用いて，最初のフレームからbboxを獲得
### ファインチューニング済みのSAM2に最初のフレームのbboxをプロンプトとして付与
### SAM2を用いてセグメンテーション
tracking_player_sam2.py を実行．引数は num_video（動画番号）, make_mask(store_true), num_prompt（何フレームおきにプロンプト与えるか）

### ファインチューニング済みのYOLOv10でトラッキング
tracking_player_yolo.py を実行．引数は num_video（動画番号

## 注意点
- まだ全て引数で与えられていないファイルがあります．使うときは黒田に連絡してください．
- ファインチューニング前のモデルはbasketballディレクトリ直下においてください．
- [sam2](https://github.com/facebookresearch/sam2)をクローンする必要があります．