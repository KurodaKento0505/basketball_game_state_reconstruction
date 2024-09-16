import json
import numpy as np
import cv2
import os
import pandas as pd
import torch
import torch.nn.utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import random
from scipy.ndimage import label, center_of_mass
from ultralytics import SAM
from pycocotools import mask as mask_utils  # RLEデコード用
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sklearn.metrics import jaccard_score
# from sklearn.model_selection import train_test_split

from evaluate_sam2 import show_prompt
from fine_tuning_sam2 import read_batch

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

def get_points(mask, num_points):  # Sample points inside the input mask
    points = []
    coords = np.argwhere(mask > 128)
    for i in range(num_points):
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([yx[1], yx[0]])
    return np.array(points)

# IoUを計算する関数
def compute_iou(pred_mask, true_mask):
    # Flatten the masks to one-dimensional arrays for IoU calculation
    pred_mask_flat = pred_mask.flatten()
    true_mask_flat = true_mask.flatten()
    # IoU (Intersection over Union) を計算
    iou = jaccard_score(true_mask_flat, pred_mask_flat, average='binary')
    return iou

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

if __name__ == '__main__':

    # train_data に image と mask を保存
    test_data = []

    # 対象のディレクトリを指定
    mask_dir = 'basketball-instants-dataset\\test_mask'
    image_dir = 'basketball-instants-dataset\\test_image'

    # ディレクトリ内のファイルを順に参照するループ
    for mask_filename in os.listdir(mask_dir):
        image_file_path = os.path.join(image_dir, mask_filename)
        mask_file_path = os.path.join(mask_dir, mask_filename)
        # Append image and corresponding mask paths
        test_data.append({
            "image": image_file_path,
            "annotation": mask_file_path
        })

    # Load the model
    checkpoint = "sam2_hiera_base_plus.pt"
    model_cfg = "sam2_hiera_b+.yaml"
    sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")

    # Build net and load weights
    predictor = SAM2ImagePredictor(sam2_model)

    # Load the fine-tuned model
    # FINE_TUNED_MODEL_WEIGHTS = "fine_tuned_sam2_3000.torch"
    # predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))

    # 全テストデータに対するIoUを計算
    total_iou = 0
    count = 0

    for data_entry in test_data:
        image_path = data_entry['image']
        mask_path = data_entry['annotation']

        # Load image, mask and point
        image, annotated_mask, input_points, num_masks = read_image(image_path, mask_path)
        input_points = np.array(input_points)
        # input_label = np.array([1])
        # show_prompt(image, input_points, input_label)

        # Skip if no points were found
        if input_points.size == 0:
            continue

        # Perform inference and predict masks
        with torch.no_grad():
            predictor.set_image(image)
            predicted_masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=np.ones([input_points.shape[0]])
            )
        
        # 最もIoUのスコアが高かったマスクを記録
        best_IoU = -1
        for k in range(len(predicted_masks)):
            iou_score = compute_iou(annotated_mask, predicted_masks[k])
            if iou_score > best_IoU:
                best_IoU = iou_score
                best_mask = predicted_masks[k]

        total_iou += best_IoU
        count += 1

        print(count, best_IoU, image_path, mask_path)
    
    mean_iou = total_iou / count
    print(f"Mean IoU across all test images: {mean_iou:.4f}")