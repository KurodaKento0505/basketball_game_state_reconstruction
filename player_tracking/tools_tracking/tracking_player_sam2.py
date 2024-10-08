import os
import gc
import cv2
import json
from tqdm import tqdm
import torch
import argparse
import pickle
from PIL import Image
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
    parser.add_argument('--make_mask', action='store_true')
    # parser.add_argument('--num_image')
    # parser.add_argument('--num_time')
    return parser.parse_args()

def initialize_sam2_model(sam2_checkpoint, model_cfg):
    # Load the SAM2 model
    sam2_model = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda")
    # if num_image > 0:
    fine_tuned_weights = f"player_segmentation/fine_tuned_sam2/1000/fine_tuned_sam2_best_step.torch"
    sam2_model.load_state_dict(torch.load(fine_tuned_weights, map_location=torch.device('cuda:1')))
    sam2_model.to(device)
    sam2_model.eval()
    return sam2_model

def run_yolo_on_frame(model, frame):
    """YOLOでフレーム上の選手を検出し、バウンディングボックスを取得"""
    results = model.track(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 選手の検出ボックス
    confidences = results[0].boxes.conf.cpu().numpy()  # confidenceの取得
    yolo_ids = results[0].boxes.id.int().cpu().numpy()
    return boxes, confidences, yolo_ids

def draw_boxes_on_frame(frame, boxes, confidences, yolo_ids):
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

def show_mask(mask, frame, color=(0, 255, 0), alpha=0.6):
    """Draws a mask on the frame with the given color and alpha transparency."""
    overlay = frame.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(mask == 1, color[c], overlay[:, :, c])
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def show_points(coords, labels, frame, marker_size=10, pos_color=(0, 255, 0), neg_color=(0, 0, 255)):
    """Draws points on the frame."""
    for coord, label in zip(coords, labels):
        color = pos_color if label == 1 else neg_color
        cv2.drawMarker(frame, tuple(coord), color, markerType=cv2.MARKER_STAR, markerSize=marker_size, thickness=2)

def show_box(box, frame, color=(0, 255, 0)):
    """Draws a bounding box on the frame."""
    x0, y0, x1, y1 = box
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness=2)

def get_bbox(mask):
    """Get the bounding box of the mask."""
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return np.array([x_min, y_min, x_max, y_max])

def process_frames(num_video, make_mask):
    """
    指定されたディレクトリ内のフレームを処理し、20フレームおきにYOLOを使用してプロンプトを取得し、
    それ以外のフレームではSAM2にプロンプトを与えずに物体追跡を行います。
    """

    INPUT_VIDEO_PATH = f'basketball-video-dataset/video/{num_video}.mp4'
    INPUT_VIDEO_FRAMES_DIR_PATH = f'basketball-video-dataset/frame/{num_video}'
    OUTPUT_VIDEO_PATH = f'player_tracking/predict_video/predict_{num_video}.mp4'
    OUTPUT_VIDEO_FRAMES_DIR_PATH = f'player_tracking/predict_frame_sam2/{num_video}'  # 処理後フレームの保存ディレクトリ
    PROCESSED_YOLO_FRAMES_PATH = f'player_detection/predict_frame/{num_video}'

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(INPUT_VIDEO_FRAMES_DIR_PATH)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    # フレーム作成
    frames_generator = sv.get_video_frames_generator(INPUT_VIDEO_PATH)
    sink = sv.ImageSink(
        target_dir_path=INPUT_VIDEO_FRAMES_DIR_PATH,
        image_name_pattern="{:05d}.jpg")
    
    with sink:
        for frame in frames_generator:
            sink.save_image(frame)

    if make_mask:
        # YOLOモデルのロード
        yolo_model = YOLO('/root/basketball/player_detection/fine_tuning/223/223_1/weights/best.pt')
        yolo_model.to(device)

        # SAM2モデルのロード
        sam2_checkpoint = "sam2_hiera_base_plus.pt"  # 適切なチェックポイントを指定
        model_cfg = "sam2_hiera_b+.yaml"  # 適切な設定ファイルを指定
        sam2_model = initialize_sam2_model(sam2_checkpoint, model_cfg)

        inference_state = sam2_model.init_state(INPUT_VIDEO_FRAMES_DIR_PATH)
        sam2_model.reset_state(inference_state)

        # プロンプト
        prompt_frame_idx = 0  # 初期フレーム
        obj_id = 1  # オブジェクトID
        while prompt_frame_idx < 10:
            prompt_frame = cv2.imread(os.path.join(INPUT_VIDEO_FRAMES_DIR_PATH, f"{prompt_frame_idx:05d}.jpg"))
            boxes, confidences, yolo_ids = run_yolo_on_frame(yolo_model, prompt_frame) # .to(device)
            yolo_first_frame = draw_boxes_on_frame(prompt_frame, boxes, confidences, yolo_ids)
            cv2.imwrite(os.path.join(PROCESSED_YOLO_FRAMES_PATH, f"{prompt_frame_idx:05d}.jpg"), yolo_first_frame)
            for box, yolo_id in zip(boxes, yolo_ids):
                _, out_obj_ids, out_mask_logits = sam2_model.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=prompt_frame_idx,
                    obj_id=yolo_id, # ann_obj_id
                    box=box,
                )
                obj_id += 1
            # 不要な変数を削除
            del prompt_frame, boxes, confidences, yolo_ids, yolo_first_frame, out_obj_ids, out_mask_logits
            gc.collect()
            torch.cuda.empty_cache()
            prompt_frame_idx += 150

        video_info = sv.VideoInfo.from_video_path(INPUT_VIDEO_PATH)
        frames_paths = sorted(sv.list_files_with_extensions(
            directory=INPUT_VIDEO_FRAMES_DIR_PATH, 
            extensions=["jpg"]))
        
        video_segments = {}  # video_segments contains the per-frame segmentation results
        with sv.VideoSink(OUTPUT_VIDEO_PATH, video_info=video_info) as sink:
            for out_frame_idx, out_obj_ids, out_mask_logits in sam2_model.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
        # video_segmentsをpickle形式で保存
        with open('/root/basketball/player_tracking/predict_video/video_segments_yolo_id_' + str(num_video) + '.pkl', 'wb') as f:
            pickle.dump(video_segments, f)
            
    else:
        # video_segmentsをpickle形式から読み込む
        with open('/root/basketball/player_tracking/predict_video/video_segments_yolo_id_' + str(num_video) + '.pkl', 'rb') as f:
            video_segments = pickle.load(f)
        # Render the segmentation results for all frames and write to video
        frame_width, frame_height = Image.open(os.path.join(INPUT_VIDEO_FRAMES_DIR_PATH, frame_names[0])).size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30.0, (frame_width, frame_height))
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0), (0, 128, 128), (128, 128, 128), (255, 165, 0), (255, 192, 203), (165, 42, 42), (173, 216, 230), (50, 205, 50)]

        for out_frame_idx in tqdm(range(len(frame_names)), desc="Processing frames"):
            frame = np.array(Image.open(os.path.join(INPUT_VIDEO_FRAMES_DIR_PATH, frame_names[out_frame_idx])))
            if out_frame_idx in video_segments:
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    show_mask(out_mask, frame, color=colors[out_obj_id])
                    
                    if out_mask.any():
                        box = get_bbox(out_mask[0])
                        show_box(box, frame, color=colors[out_obj_id])
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Release the video writer
    out.release()

if __name__ == '__main__':
    args = parse_arguments()
    # num_images = [int(num_image) for num_image in args.num_image.split(",")]
    num_video = args.num_video
    make_mask = args.make_mask
    # for num_image in num_images:
    segmentation_results = process_frames(num_video, make_mask)
