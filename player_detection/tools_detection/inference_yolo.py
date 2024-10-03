from ultralytics import YOLO
import argparse
import json
import os
import cv2
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import numpy as np
# from sklearn.metrics import jaccard_score  # IoU計算のため

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_image')
    parser.add_argument('--num_time')
    return parser.parse_args()

def get_image_shape(image):
    image_height, image_width = image.shape[:2]
    return image_height, image_width

def calculate_pairwise_iou(pred_boxes, true_boxes, image_height, image_width):
    """予測と正解ボックス間のペアワイズIoUを計算し、最適なマッチングを行う"""
    
    def box_iou(box1, box2):
        """2つのバウンディングボックスのIoUを計算する関数"""
        # box1は左上と右下の座標 (x_min, y_min, x_max, y_max)
        x1_min, y1_min, x1_max, y1_max = box1
        x1_min, y1_min, x1_max, y1_max = x1_min/image_width, y1_min/image_height, x1_max/image_width, y1_max/image_height
        
        # box2は中心点の座標と幅・高さ (cx, cy, w, h)
        cx, cy, w, h = box2
        
        # box2を左上と右下の座標に変換
        x2_min = cx - w / 2
        y2_min = cy - h / 2
        x2_max = cx + w / 2
        y2_max = cy + h / 2
        
        # 交差部分の座標を計算
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        # 交差領域の面積を計算
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        
        # 各ボックスの面積を計算
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        # IoUを計算
        if (box1_area + box2_area - inter_area) == 0:
            return 0
        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou
    
    # ペアワイズIoU行列を計算
    num_preds = len(pred_boxes)
    num_trues = len(true_boxes)
    iou_matrix = np.zeros((num_preds, num_trues))

    for i, pred in enumerate(pred_boxes):
        for j, true in enumerate(true_boxes):
            iou_matrix[i, j] = box_iou(pred, true)

    # ハンガリアンアルゴリズムで最適マッチングを計算 (線形割り当て問題)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # マイナスを付けて最大値を最適化

    # マッチング結果のIoUを抽出
    matched_ious = iou_matrix[row_ind, col_ind]
    
    # 全体の平均IoUを計算
    mean_iou = np.mean(matched_ious) if len(matched_ious) > 0 else 0
    return mean_iou, matched_ious


def inference_yolo(num_image, num_time):
    # モデルのロード
    if num_image == 0:
        model = YOLO('/root/basketball/yolov10n.pt')
    else:
        model = YOLO('/root/basketball/player_detection/fine_tuning/' + str(num_image) + '/' + str(num_image) + '_' + str(num_time) + '/weights/best.pt')

    # valid用の画像ディレクトリ
    valid_image_dir = Path(f'/root/basketball/player_detection/dataset_yolo/10/valid/images')
    # 正解データのディレクトリ (アノテーション)
    valid_label_dir = Path(f'/root/basketball/player_detection/dataset_yolo/10/valid/labels')
    
    # 画像とアノテーションをペアで取得
    image_files = list(valid_image_dir.glob('*.png'))
    iou_results = []
    all_ious = []

    for image_file in image_files:
        # 画像に対する推論
        results = model(image_file)

        # 画像の幅と高さを取得
        image = cv2.imread(image_file)
        image_height, image_width = get_image_shape(image)
        
        # 推論結果のバウンディングボックス取得
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()  # 予測ボックス
        # 画像に対応する正解データをロード (YOLO形式)
        label_file = valid_label_dir / f'{image_file.stem}.txt'
        true_boxes = []
        
        with open(label_file, 'r') as f:
            for line in f.readlines():
                label_data = line.split()
                true_boxes.append([float(i) for i in label_data[1:]])  # ラベルID以外の値がボックス
        
        # IoUを計算
        iou, _ = calculate_pairwise_iou(pred_boxes, true_boxes, image_height, image_width)
        all_ious.append(iou)

        # 結果を保存
        iou_results.append({
            'image': image_file.name,
            'iou': iou
        })

    # JSONファイルに保存
    with open('player_detection/fine_tuning/' + str(num_image) + '/iou_results_' + str(num_time) + '.json', 'w') as f:
        json.dump(iou_results, f, indent=4)
    
    # 全体の平均IoUを計算
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0
    return mean_iou

if __name__ == '__main__':
    args = parse_arguments()
    num_times = [int(num_time) for num_time in args.num_time.split(",")]
    num_images = [int(num_image) for num_image in args.num_image.split(",")]
    for num_time in num_times:
        all_results = []
        for num_image in num_images:
            mean_iou = inference_yolo(num_image, num_time)
            all_results.append({
                "num_image": num_image,
                "mean_iou": mean_iou
            })
        with open('player_detection/fine_tuning/all/iou_results_' + str(num_time) + '.json', 'w') as f:
            json.dump(all_results, f, indent=4)