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
from fine_tuning_sam2 import get_points, SAMDataset

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_image')
    parser.add_argument('--num_train_step', default='best')
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

def inference(num_image, num_train_step, dataset):

    # Load the model
    # Load the model configuration
    model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # Load the fine-tuned model
    # Create an instance of the model architecture with the loaded configuration
    my_model = SamModel(config=model_config)
    #Update the model by loading the weights from saved file.
    my_model.load_state_dict(torch.load("basketball-player-segmentation/fine_tuned_model/" + str(num_image) + "/fine_tuned_sam2_" + num_train_step + ".torch"))

    # set the device to cuda if available, otherwise use cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    my_model.to(device)

    # 全テストデータに対するIoUを計算
    total_iou = 0
    count = 0

    for idx in range(len(dataset)):
        # load image
        test_image = dataset[idx]["image"]
        # get box prompt based on ground truth segmentation map
        ground_truth_mask = np.array(dataset[idx]["label"])
        prompt = dataset[idx]["prompt"]
        # prepare image + box prompt for the model
        inputs = processor(test_image, input_points=[prompt], return_tensors="pt")
        # Move the input tensor to the GPU if it's not already there
        inputs = {k: v.to(device) for k, v in inputs.items()}
        my_model.eval()

        # forward pass
        with torch.no_grad():
            outputs = my_model(**inputs, multimask_output=False)

        # apply sigmoid
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        union = gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter
        iou = inter / union
        current_mean_iou = np.mean(iou.cpu().detach().numpy())
        print('inter:', inter)
        print('union:', union)
        print('current_mean_iou:', current_mean_iou)

if __name__ == '__main__':
    args = parse_arguments()
    num_train_step = args.num_train_step
    num_images = [int(num_image) for num_image in args.num_image.split(",")]
    
    # 対象のディレクトリを指定
    mask_dir = 'basketball-instants-dataset//train_mask'
    image_dir = 'basketball-instants-dataset//train_image'
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
        save_shape = (256, 256)
        image = cv2.resize(image, save_shape)
        image = np.transpose(image, (2, 0, 1))
        mask = cv2.resize(mask, save_shape)
        prompt = get_points(mask, num_points=3)
        images.append(image)
        masks.append(mask)
        prompts.append(prompt)
        count += 1
    # すべての画像を numpy の三次元配列に結合
    stacked_images = np.stack(images)
    stacked_masks = np.stack(masks)
    stacked_prompts = np.stack(prompts)
    # Convert the NumPy arrays to Pillow images and store them in a dictionary
    dataset_dict = {
        "image": [img for img in stacked_images],
        "label": [mask for mask in stacked_masks],
        "prompt": [prompt for prompt in stacked_prompts]
    }
    # Create the dataset using the datasets.Dataset class
    dataset = Dataset.from_dict(dataset_dict)
    
    for num_image in num_images:
        print("\n", num_image)
        inference(num_image, num_train_step, dataset)