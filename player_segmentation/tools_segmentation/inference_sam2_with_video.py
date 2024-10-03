import time
import argparse
import os
import cv2
import torch
import pickle
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
# from natsort import natsorted
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda:1", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_image')
    parser.add_argument('--num_train_step', default='best')
    parser.add_argument('--model')
    return parser.parse_args()

# マスク描画用関数
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# プロンプト描画用関数
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def initialize_sam2_model(sam2_checkpoint, model_cfg, num_image, num_train_step):
    # Load the SAM2 model
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda")
    if num_image > 0:
        fine_tuned_weights = f"fine_tuned_model/{num_image}/fine_tuned_sam2_{num_train_step}_step.torch"
        predictor.load_state_dict(torch.load(fine_tuned_weights))
    return predictor

def detect_players_with_yolo(frame, yolo_model):
    # Run YOLO to detect players in the frame
    results = yolo_model.predict(frame)  # Results object
    player_boxes = []
    # YOLOv5's results provide .boxes for bounding boxes and .names for labels
    boxes = results[0].boxes  # List of bounding boxes in the first image (frame)
    for box in boxes:
        cls = int(box.cls)  # Class ID
        label = yolo_model.names[cls]  # Class label using YOLO's internal label map
        if label == 'person':  # 'person' class in YOLO represents players
            # Convert to [x1, y1, x2, y2] format
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            player_boxes.append([x1, y1, x2, y2])  # Store the bounding box coordinates
    return player_boxes

def extract_points_from_boxes(player_boxes):
    # Convert bounding boxes into center points and labels for SAM2
    points = []
    for box in player_boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        points.append([cx, cy])
    points = np.array(points, dtype=np.float32)
    labels = np.ones(len(points), dtype=np.int32)  # All labels as 1 (positive)
    return points, labels

def save_data(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def inference_with_video(num_image, num_train_step, model, force_recompute=False):
    frame_dir_path = "basketball-instants-dataset/frame/2"
    predict_frame_dir_path = "basketball-instants-dataset/predict_frame/" + model + "/2"
    output_video_path = "basketball-instants-dataset/predict_video/2/output.mp4"
    frame_cache_path = "basketball-instants-dataset/predict_video/" + model + "/2/frame_cache.pkl"
    segments_cache_path = "basketball-instants-dataset/predict_video/" + model + "/2/video_segments_cache.pkl"

    # キャッシュからフレーム名リストを取得
    if not os.path.exists(frame_cache_path) or force_recompute:
        frame_names = [
            p for p in os.listdir(frame_dir_path)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        save_data(frame_cache_path, frame_names)
    else:
        frame_names = load_data(frame_cache_path)

    # 推論済みのセグメンテーション結果をキャッシュから読み込み
    if not os.path.exists(segments_cache_path) or force_recompute:
        # モデルのロード
        sam2_checkpoint = "sam2_hiera_base_plus.pt"
        model_cfg = "sam2_hiera_b+.yaml"
        predictor = initialize_sam2_model(sam2_checkpoint, model_cfg, num_image, num_train_step)

        inference_state = predictor.init_state(video_path=frame_dir_path)
        predictor.reset_state(inference_state)

        ann_frame_idx = 0  # 初期フレーム
        ann_obj_id = 1  # オブジェクトID

        # 最初のプロンプト
        points = np.array([[332, 356],[336, 417],[325, 446],[305, 487],[330, 321],[351, 479],[354, 521],[293, 521],[294, 360]], dtype=np.float32)
        labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        # 結果のプロパゲート（伝播）を実行
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # 推論結果をキャッシュに保存
        save_data(segments_cache_path, video_segments)
    else:
        video_segments = load_data(segments_cache_path)

    # ビデオ生成のための処理
    frame_sample = Image.open(os.path.join(frame_dir_path, frame_names[0]))
    height, width = frame_sample.size[1], frame_sample.size[0]
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    # セグメンテーション結果をビデオに書き込む
    for out_frame_idx in range(len(frame_names)):
        plt.close("all")
        frame = Image.open(os.path.join(frame_dir_path, frame_names[out_frame_idx]))
        frame_np = np.array(frame)

        for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():
            # マスクを描画
            mask = (out_mask > 0.5).astype(np.uint8)
            mask_colored = np.zeros_like(frame_np)
            mask = np.transpose(mask, (1, 2, 0))
            mask_colored[:,:,1][mask[:,:,0] == 1] = 255  # 緑色

            # フレームにマスクをオーバーレイ
            frame_np = cv2.addWeighted(frame_np, 1, mask_colored, 0.5, 0)

        # フレームを書き込む
        video_writer.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

        # フレームを保存
        predict_frame_path = os.path.join(predict_frame_dir_path, f"{frame_names[out_frame_idx]}")
        Image.fromarray(frame_np).save(predict_frame_path)


    video_writer.release()
    print(f"Output video saved at {output_video_path}")


if __name__ == '__main__':
    args = parse_arguments()
    num_train_step = args.num_train_step
    model = args.model
    num_images = [int(num_image) for num_image in args.num_image.split(",")]
    for num_image in num_images:
        print("\n", num_image)
        inference_with_video(num_image, num_train_step, model)