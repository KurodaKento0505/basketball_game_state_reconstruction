import os
import gc
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
    parser.add_argument('--num_video')
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

def process_frames(num_video):
    """
    指定されたディレクトリ内のフレームを処理し、20フレームおきにYOLOを使用してプロンプトを取得し、
    それ以外のフレームではSAM2にプロンプトを与えずに物体追跡を行います。
    """

    SOURCE_VIDEO_PATH = f'basketball-video-dataset/video/{num_video}.mp4'
    VIDEO_FRAMES_DIRECTORY_PATH = f'basketball-video-dataset/frame/{num_video}'
    TARGET_VIDEO_PATH = f'player_tracking/predict_video/predict_{num_video}.mp4'
    PROCESSED_FRAMES_PATH = f'player_tracking/predict_frame_sam2/{num_video}'  # 処理後フレームの保存ディレクトリ
    PROCESSED_YOLO_FRAMES_PATH = f'player_detection/predict_frame/{num_video}'

    # YOLOモデルのロード
    yolo_model = YOLO('/root/basketball/player_detection/fine_tuning/223/223_1/weights/best.pt')
    yolo_model.to(device)

    # SAM2モデルのロード
    sam2_checkpoint = "sam2_hiera_base_plus.pt"  # 適切なチェックポイントを指定
    model_cfg = "sam2_hiera_b+.yaml"  # 適切な設定ファイルを指定
    sam2_model = initialize_sam2_model(sam2_checkpoint, model_cfg)

    frames_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    sink = sv.ImageSink(
        target_dir_path=VIDEO_FRAMES_DIRECTORY_PATH,
        image_name_pattern="{:05d}.jpg")
    
    with sink:
        for frame in frames_generator:
            sink.save_image(frame)

    inference_state = sam2_model.init_state(VIDEO_FRAMES_DIRECTORY_PATH)
    sam2_model.reset_state(inference_state)

    colors = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
    mask_annotator = sv.MaskAnnotator(
        color=sv.ColorPalette.from_hex(colors),
        color_lookup=sv.ColorLookup.TRACK)

    # 最初のプロンプト
    ann_frame_idx = 0  # 初期フレーム
    ann_obj_id = 1  # オブジェクトID
    while ann_frame_idx < 508:
        first_frame = cv2.imread(os.path.join(VIDEO_FRAMES_DIRECTORY_PATH, f"{ann_frame_idx:05d}.jpg"))
        boxes, confidences = run_yolo_on_frame(yolo_model, first_frame) # .to(device)
        yolo_first_frame = draw_boxes_on_frame(first_frame, boxes, confidences)
        cv2.imwrite(os.path.join(PROCESSED_YOLO_FRAMES_PATH, f"{ann_frame_idx:05d}.jpg"), yolo_first_frame)
        for box in boxes:
            _, out_obj_ids, out_mask_logits = sam2_model.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box,
            )
            ann_obj_id += 1
        # 不要な変数を削除
        del first_frame, boxes, confidences, yolo_first_frame, out_obj_ids, out_mask_logits
        gc.collect()
        torch.cuda.empty_cache()
        ann_frame_idx += 10

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    frames_paths = sorted(sv.list_files_with_extensions(
        directory=VIDEO_FRAMES_DIRECTORY_PATH, 
        extensions=["jpg"]))

    with sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info) as sink:
        for frame_idx, object_ids, mask_logits in sam2_model.propagate_in_video(inference_state):
            frame = cv2.imread(frames_paths[frame_idx])
            # frame_tensor = torch.from_numpy(frame).to(device)
            masks = (mask_logits > 0.0).cpu().numpy()
            N, X, H, W = masks.shape
            masks = masks.reshape(N * X, H, W)
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks,
                tracker_id=np.array(object_ids)
            )
            frame = mask_annotator.annotate(frame, detections)
            # フレームを個別に保存
            processed_frame_path = os.path.join(PROCESSED_FRAMES_PATH, f"{frame_idx:05d}.jpg")
            cv2.imwrite(processed_frame_path, frame)
            # 不要な変数を削除
            del frame, masks, detections
            gc.collect()
            torch.cuda.empty_cache()
            # フレームを動画としても保存
            # sink.write_frame(frame)

if __name__ == '__main__':
    args = parse_arguments()
    # num_images = [int(num_image) for num_image in args.num_image.split(",")]
    num_video = args.num_video
    # for num_image in num_images:
    segmentation_results = process_frames(num_video)
