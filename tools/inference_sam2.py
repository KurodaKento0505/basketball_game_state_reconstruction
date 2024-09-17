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
from fine_tuning_sam2 import parse_arguments

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

def inference(num_image, num_train_step, image, mask, input_point, num_masks):

    # Load the model
    checkpoint = "sam2_hiera_base_plus.pt"
    model_cfg = "sam2_hiera_b+.yaml"
    sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")

    # Build net and load weights
    predictor = SAM2ImagePredictor(sam2_model)

    # Load the fine-tuned model
    model_path = "fine_tuned_model//" + str(num_image) + "//fine_tuned_sam2_"+ str(num_train_step) + ".torch"
    predictor.model.load_state_dict(torch.load(model_path))

    # 全テストデータに対するIoUを計算
    total_iou = 0
    count = 0

    NO_OF_STEPS = 1

    for step in range(1, NO_OF_STEPS + 1):
        with torch.cuda.amp.autocast():
            if image is None or mask is None or num_masks == 0:
                continue

            input_label = np.ones((num_masks, 1))
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                continue

            if input_point.size == 0 or input_label.size == 0:
                continue

            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                continue

            # プロンプト埋め込みの生成
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None,
            )

            # 特徴量とマスクの予測
            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            # 予測マスクの後処理
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
            print('prd_masks:', prd_masks.shape)
            # 正解マスクと予測マスクの準備
            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])

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
    image, mask, input_point, num_masks = read_batch(test_data, visualize_data=False)
    for num_image in num_images:
        print("\n", num_image)
        inference(num_image, num_train_step, image, mask, input_point, num_masks)