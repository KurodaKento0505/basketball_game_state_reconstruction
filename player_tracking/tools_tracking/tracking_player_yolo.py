import os
import cv2
import json
import torch
import argparse
from ultralytics import YOLO
import supervision as sv
# from segment_anything import sam_model_registry, SamPredictor
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.sam2_video_predictor import SAM2VideoPredictor
import numpy as np
from pathlib import Path

# 使用するデバイスをcuda:1に設定
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_video')
    # parser.add_argument('--num_time')
    return parser.parse_args()

def run_yolo_on_frame(model, frame):
    """YOLOでフレーム上の選手を検出し、バウンディングボックスを取得"""
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 選手の検出ボックス
    confidences = results[0].boxes.conf.cpu().numpy()  # confidenceの取得
    return boxes, confidences

def draw_boxes_on_frame(frame, boxes, confidences):
    """フレームにバウンディングボックスと信頼度を描画する関数"""
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box[:4])  # 座標を整数に変換
        confidence = confidences[i]  # 信頼度を取得
        color = (0, 255, 0)  # バウンディングボックスの色（緑）
        thickness = 2  # 線の太さ
        # バウンディングボックスを描画
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
        # 信頼度を描画（小数点以下2桁にフォーマット）
        label = f'{confidence:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # フォントの大きさ
        font_thickness = 1  # フォントの太さ
        text_color = (255, 0, 0)  # テキストの色（青）
        # テキストを描画（バウンディングボックスの左上に）
        cv2.putText(frame, label, (x_min, y_min - 5), font, font_scale, text_color, font_thickness)
    return frame

def process_frames(num_video):
    """
    動画の各フレームを処理し、YOLOの検出結果をJSONファイルにまとめて保存します。
    """

    SOURCE_VIDEO_PATH = f'basketball-video-dataset/video/{num_video}.mp4'
    DETECTION_RESULTS_PATH = f'player_tracking/predict_frame_yolo/{num_video}.json'  # 検出結果の保存パス

    # YOLOモデルのロード
    yolo_model = YOLO('/root/basketball/player_detection/fine_tuning/223/223_1/weights/best.pt')
    yolo_model.to(device)
    # Open the video file
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    frame_count = 0
    # 全フレームの検出結果を格納するリスト
    all_detections = []

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # YOLOを実行して結果を取得
            results = yolo_model.track(frame, persist=True)  # iou=0.8, conf=0.7, persist=True
            # 検出結果を取得
            boxes = results[0].boxes.xyxy.cpu().numpy()  # 検出ボックス
            confidences = results[0].boxes.conf.cpu().numpy()  # confidenceの取得
            yolo_ids = results[0].boxes.id.int().cpu().numpy()  # IDの取得
            # フレームごとの検出結果をリストに追加
            frame_detections = {
                "frame": frame_count,
                "detections": []
            }
            for box, conf, yolo_id in zip(boxes, confidences, yolo_ids):
                detection = {
                    "box": {
                        "x_min": float(box[0]),
                        "y_min": float(box[1]),
                        "x_max": float(box[2]),
                        "y_max": float(box[3])
                    },
                    "confidence": float(conf),
                    "id": int(yolo_id)
                }
                frame_detections["detections"].append(detection)
            all_detections.append(frame_detections)
        else:
            # 動画の終わりに到達した場合
            break
        frame_count += 1

    # 動画全体の検出結果をJSONファイルに保存
    with open(DETECTION_RESULTS_PATH, 'w') as json_file:
        json.dump(all_detections, json_file, indent=4)

    # 終了処理
    cap.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_arguments()
    # num_images = [int(num_image) for num_image in args.num_image.split(",")]
    num_video = args.num_video
    # for num_image in num_images:
    segmentation_results = process_frames(num_video)

    # 結果の処理（保存や表示）
    print("Segmentation completed for all frames.")