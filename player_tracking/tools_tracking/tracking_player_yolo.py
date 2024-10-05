import os
import cv2
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
    # parser.add_argument('--num_image')
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

def process_frames():
    """
    指定されたディレクトリ内のフレームを処理し、20フレームおきにYOLOを使用してプロンプトを取得し、
    それ以外のフレームではSAM2にプロンプトを与えずに物体追跡を行います。
    """

    SOURCE_VIDEO_PATH = 'basketball-video-dataset/video/1.mp4'
    VIDEO_FRAMES_DIRECTORY_PATH = 'basketball-video-dataset/frame/1'
    TARGET_VIDEO_PATH = 'player_tracking/predict_video/predict_1.mp4'
    PROCESSED_FRAMES_PATH = 'player_tracking/predict_frame_yolo/1'  # 処理後フレームの保存ディレクトリ

    # YOLOモデルのロード
    yolo_model = YOLO('/root/basketball/player_detection/fine_tuning/223/223_1/weights/best.pt') # 
    yolo_model.to(device)

    # Open the video file
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

    # Loop through the video frames
    frame_count = 0
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = yolo_model.track(frame, conf=0.7, persist=True) # iou=0.8, 

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imwrite(os.path.join(PROCESSED_FRAMES_PATH, f"{frame_count:05d}.jpg"), annotated_frame)

            '''# Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break'''
        else:
            # Break the loop if the end of the video is reached
            break
        frame_count += 1

    # Release the video capture object and close the display window
    cap.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_arguments()
    # num_images = [int(num_image) for num_image in args.num_image.split(",")]
    video_frame_dir = "/root/basketball/basketball-video-dataset/frame/1"  # 動画ファイルのパスを指定
    output_dir = "player_tracking/predict_frame"
    # for num_image in num_images:
    segmentation_results = process_frames()

    # 結果の処理（保存や表示）
    print("Segmentation completed for all frames.")