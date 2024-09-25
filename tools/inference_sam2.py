import json
import numpy as np
import cv2
import os
import pandas as pd
import torch
import torch.nn.utils
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import random
from scipy.ndimage import label, center_of_mass
from ultralytics import SAM
from pycocotools import mask as mask_utils  # RLEデコード用
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import SamModel, SamConfig, SamProcessor
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
# from sklearn.model_selection import train_test_split

from evaluate_sam2 import show_prompt
from fine_tuning_sam2 import get_points, SAMDataset, read_batch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_image')
    parser.add_argument('--num_train_step', default='best')
    parser.add_argument('--num_time')
    return parser.parse_args()

def read_image(image_path, mask_path):  # read and resize image and mask
    # Get full paths
    Img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
    ann_map = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read annotation as grayscale
    if Img is None or ann_map is None:
        print(f"Error: Could not read image or mask from path {image_path} or {mask_path}")
        return None, None, None, 0
    # Resize image and mask
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # Scaling factor
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    # Initialize a single binary mask
    binary_mask = np.zeros_like(ann_map, dtype=np.uint8)
    points = []
    # Get binary masks and combine them into a single mask
    inds = np.unique(ann_map)[1:]  # Skip the background (index 0)
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)  # Create binary mask for each unique index
        binary_mask = np.maximum(binary_mask, mask)  # Combine with the existing binary mask
    # Erode the combined binary mask to avoid boundary points
    eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)
    # Get all coordinates inside the eroded mask and choose a random point
    coords = np.argwhere(eroded_mask > 0)
    if len(coords) > 0:
        for _ in inds:  # Select as many points as there are unique labels
            yx = np.array(coords[np.random.randint(len(coords))])
            points.append([yx[1], yx[0]])

    return Img, binary_mask, points, len(inds)

def overlay_masks(image, prd_mask, gt_mask, alpha=0.5):
    """
    予測マスクとGTマスクを元画像に重ね合わせた画像を生成
    """
    # 予測マスクとGTマスクをRGBに変換
    prd_mask_colored = np.zeros_like(image)
    gt_mask_colored = np.zeros_like(image)
    
    prd_mask_colored[:, :, 0] = prd_mask[0] * 255  # Red for predicted mask
    gt_mask_colored[:, :, 1] = gt_mask * 255    # Green for ground truth mask

    # 重ね合わせる
    overlay = cv2.addWeighted(prd_mask_colored, alpha, gt_mask_colored, alpha, 0)
    result_image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    return result_image

# IoUの計算を行う関数
def calculate_iou(pred_mask, true_mask):
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    true_mask = true_mask.astype(np.uint8)
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 1.0  # 両方とも背景の場合はIoU=1
    return intersection / union

def show_predicted_image(predicted_masks, scores, image):
    # Process the predicted masks and sort by scores
    np_masks = np.array(predicted_masks[:, 0])
    np_scores = scores[:, 0]
    sorted_masks = np_masks[np.argsort(np_scores)][::-1]

    # Initialize segmentation map and occupancy mask
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)

    # Combine masks to create the final segmentation map
    for i in range(sorted_masks.shape[0]):
        mask = sorted_masks[i]
        if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
            continue

        mask_bool = mask.astype(bool)
        mask_bool[occupancy_mask] = False  # Set overlapping areas to False in the mask
        seg_map[mask_bool] = i + 1  # Use boolean mask to index seg_map
        occupancy_mask[mask_bool] = True  # Update occupancy_mask

    # Visualization: Show the original image, mask, and final segmentation side by side
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title('Test Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Original Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Final Segmentation')
    plt.imshow(seg_map, cmap='jet')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def inference(num_image, num_train_step, num_time, test_data):
    # load the model
    sam2_checkpoint = "sam2_hiera_base_plus.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
    model_cfg = "sam2_hiera_b+.yaml" # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    # Build net and load weights
    predictor = SAM2ImagePredictor(sam2_model)
    if num_image == 0:
        pass
    else:
        FINE_TUNED_MODEL_WEIGHTS = "fine_tuned_model//" + str(num_image) + "//fine_tuned_sam2_" + num_train_step + "_step.torch"
        predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))

    #Training loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model.to(device)
    sam2_model.eval()

    print('--start inference--')

    iou_list = []
    results = []
    for i in range(len(test_data)):
        with torch.cuda.amp.autocast():
            image, mask, input_point, num_masks = read_batch(test_data[i], visualize_data=False)

            # input_pointが1次元の場合、2次元に変換
            if len(input_point.shape) != 3:
                continue
            
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=np.ones((input_point.shape[0], 1))
            )

            # 予測マスクとGTマスクのIoUを計算
            prd_mask = torch.sigmoid(torch.tensor(masks[0])).cpu().detach().numpy()  # CPUに移動
            gt_mask = torch.tensor(mask.astype(np.float32)).cpu().detach().numpy()  # CPUに移動
            intersection = (prd_mask > 0.5).astype(np.float32) * gt_mask
            union = (prd_mask > 0.5).astype(np.float32) + gt_mask - intersection
            iou = intersection.sum() / union.sum()
            print(iou)

            # 1〜5枚目の画像を保存
            if i < 5:
                # 予測マスクと正解マスクを元画像に重ね合わせ
                overlaid_image = overlay_masks(image, prd_mask > 0.5, gt_mask)

                # 画像を保存
                plt.figure(figsize=(10, 10))
                plt.imshow(overlaid_image)
                plt.axis('off')
                plt.title(f'Image {i} - IoU: {iou:.4f}')
                plt.savefig(f'visualize_inference//' + str(num_time) + '//' + str(num_image) + '//overlay_image_' + str(i + 1) + '.png')
                plt.close()
            
            # 結果をJSONファイルに保存
            results.append({
                "image_names": f"image_" + str(i + 1),
                "iou": float(iou),
            })
            iou_list.append(float(iou))

    # IoUの平均を計算
    mean_iou = np.mean(iou_list)

    ###################################
    with open('visualize_inference//' + str(num_time) + '//' + str(num_image) + '//iou_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f'Inference finished. Mean IoU: {mean_iou:.4f}')
    print('i:', i)

    return mean_iou

if __name__ == '__main__':
    args = parse_arguments()
    num_train_step = args.num_train_step
    num_time = args.num_time
    num_images = [int(num_image) for num_image in args.num_image.split(",")]
    
    # 対象のディレクトリを指定
    mask_dir = 'basketball-instants-dataset//test_mask'
    image_dir = 'basketball-instants-dataset//test_image'
    # ディレクトリ内のファイルを順に参照するループ
    count = 0
    images = []
    masks = []
    prompts = []
    for mask_filename in os.listdir(mask_dir):
        image_file_path = os.path.join(image_dir, mask_filename)
        mask_file_path = os.path.join(mask_dir, mask_filename)
        image = cv2.imread(image_file_path)  # 画像をグレースケールで読み込む例
        mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
        save_shape = (512, 512)
        image = cv2.resize(image, save_shape)
        # image = np.transpose(image, (2, 0, 1))
        mask = cv2.resize(mask, save_shape)
        images.append(image)
        masks.append(mask)
        count += 1
    # すべての画像を numpy の三次元配列に結合
    stacked_images = np.stack(images)
    stacked_masks = np.stack(masks)
    # Convert the NumPy arrays to Pillow images and store them in a dictionary
    test_data = []
    for i in range(len(stacked_images)):
        test_data.append({
       "image": stacked_images[i],
       "annotation": stacked_masks[i]
   })
    all_results = []
    for num_image in num_images:
        print("\n", num_image)
        mean_iou = inference(num_image, num_train_step, num_time, test_data)
        all_results.append({
            "num_image": num_image,
            "mean_iou": mean_iou
        })
    with open('visualize_inference//all//iou_results_' + str(num_time) + '.json', 'w') as f:
        json.dump(all_results, f, indent=4)