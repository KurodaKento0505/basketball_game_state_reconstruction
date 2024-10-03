import os
import cv2
import torch
import argparse
from ultralytics import YOLO
# from segment_anything import sam_model_registry, SamPredictor
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.sam2_video_predictor import SAM2VideoPredictor
import numpy as np
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_image')
    parser.add_argument('--num_time')
    return parser.parse_args()

def initialize_sam2_model(sam2_checkpoint, model_cfg):
    # Load the SAM2 model
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda")
    if num_image > 0:
        fine_tuned_weights = f"fine_tuned_model/1000/fine_tuned_sam2_best_step.torch"
        predictor.load_state_dict(torch.load(fine_tuned_weights))
    return predictor

def run_yolo_on_frame(model, frame):
    """YOLOでフレーム上の選手を検出し、バウンディングボックスを取得"""
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 選手の検出ボックス
    return boxes

def run_sam_on_frame(predictor, frame, boxes):
    """SAM2でフレームのセグメンテーションを実行し、バウンディングボックスをプロンプトとして与える"""
    input_boxes = np.array(boxes)  # YOLOの出力を使用
    predictor.set_image(frame)
    masks, _, _ = predictor.predict(box=input_boxes, multimask_output=False)  # SAM2でマスクを取得
    return masks

def draw_boxes_on_frame(frame, boxes):
    """フレームにバウンディングボックスを描画する関数"""
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box[:4])  # 座標を整数に変換
        color = (0, 255, 0)  # バウンディングボックスの色（緑）
        thickness = 2  # 線の太さ
        # バウンディングボックスを描画
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
    return frame


def apply_masks_to_image(image, masks):
    # マスクを適用するためにカラー化
    colored_masks = []
    for mask in masks:
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color
        colored_masks.append(colored_mask)
    # すべてのマスクを合成
    combined_mask = np.zeros_like(image)
    for colored_mask in colored_masks:
        combined_mask = cv2.add(combined_mask, colored_mask)
    # 元画像とマスクをブレンド
    alpha = 0.5  # マスクの透明度
    blended_image = cv2.addWeighted(image, 1, combined_mask, alpha, 0)
    return blended_image

def process_frames(frame_dir, output_dir):
    """
    指定されたディレクトリ内のフレームを処理し、20フレームおきにYOLOを使用してプロンプトを取得し、
    それ以外のフレームではSAM2にプロンプトを与えずに物体追跡を行います。
    """
    # YOLOモデルのロード
    yolo_model = YOLO('/root/basketball/player_detection/fine_tuning/223/223_1/weights/best.pt')

    # SAM2モデルのロード
    sam2_checkpoint = "sam2_hiera_base_plus.pt"  # 適切なチェックポイントを指定
    model_cfg = "sam2_hiera_b+.yaml"  # 適切な設定ファイルを指定
    predictor = initialize_sam2_model(sam2_checkpoint, model_cfg)

    inference_state = predictor.init_state(video_path=frame_dir)
    predictor.reset_state(inference_state)

    ann_frame_idx = 0  # 初期フレーム
    ann_obj_id = 1  # オブジェクトID

    # 最初のプロンプト
    first_frame = frame_dir + '/00001.jpg'
    boxes = run_sam_on_frame(yolo_model, first_frame)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        boxes=boxes,
    )
    # 結果のプロパゲート（伝播）を実行
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # フレームが格納されているディレクトリのパスを取得
    frames_path = Path(frame_dir)
    frames = sorted(frames_path.glob('*.jpg'))  # JPEG形式のフレームを取得（必要に応じて拡張子を変更）

    total_frames = len(frames)
    print(f"Total frames: {total_frames}")

    # 出力ディレクトリの設定
    output_dir = os.path.join(frame_dir, 'segmented_output')
    os.makedirs(output_dir, exist_ok=True)

    boxes = []  # 初期化しておく
    for current_frame, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))  # フレームを読み込む
        if frame is None:
            print(f"Failed to read frame: {frame_path}")
            continue

        # 20フレームごとにYOLOでバウンディングボックスを取得
        if current_frame % 20 == 0:
            boxes = run_yolo_on_frame(yolo_model, frame)
            print(f"Frame {current_frame}: Detected {len(boxes)} players.")

        # SAM2でセグメンテーションを行う
        masks = run_sam_on_frame(sam2_predictor, frame, boxes if current_frame % 20 == 0 else None)
        print(f"Frame {current_frame}: SAM2 completed segmentation.")

        # セグメンテーションマスクを元画像に反映
        segmented_image = apply_masks_to_image(frame, masks)

        # 結果を保存
        output_image_path = os.path.join(output_dir, frame_path.name)
        cv2.imwrite(output_image_path, segmented_image)
        print(f"Saved segmented image: {output_image_path}")

    print("Processing completed.")


if __name__ == '__main__':
    args = parse_arguments()
    num_images = [int(num_image) for num_image in args.num_image.split(",")]
    video_frame_dir = "/root/basketball/basketball-video-dataset/frame/1"  # 動画ファイルのパスを指定
    output_dir = "player_tracking/predict_frame"
    for num_image in num_images:
        segmentation_results = process_frames(video_frame_dir, output_dir, num_image)

    # 結果の処理（保存や表示）
    print("Segmentation completed for all frames.")
