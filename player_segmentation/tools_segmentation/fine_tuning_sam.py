import json
import numpy as np
import cv2
import os
import pandas as pd
import torch
import torch.nn.utils
from torch.utils.data import DataLoader
from torch.nn.functional import threshold, normalize
from tqdm import tqdm
from statistics import mean
from torch.optim import Adam
import argparse
# import tifffile
from patchify import patchify 
from scipy import ndimage
from PIL import Image
from datasets import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import label, center_of_mass
from ultralytics import SAM
from pycocotools import mask as mask_utils  # RLEデコード用
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import SamModel, SamConfig, SamProcessor
# import monai
# from sklearn.model_selection import train_test_split

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_image')
    parser.add_argument('--batch_size', type = int)
    parser.add_argument('--num_epoch', type = int)
    return parser.parse_args()

def get_points(mask, num_points):  # Sample points inside the input mask
    points = []
    coords = np.argwhere(mask > 128)
    for i in range(num_points):
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([yx[1], yx[0]])
    return np.array(points)

def dice_loss(pred, target, smooth = 10.):
    pred = torch.sigmoid(pred)
    # pred = torch.clamp(pred, 0, 1)  # 0〜1の範囲にクリップ
    # target = torch.clamp(target, 0, 1)
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def read_batch(data, visualize_data=False):
    all_images = []
    all_masks = []
    all_points = []

    for ent in data:
        # Get full paths
        Img = cv2.imread(ent["image"])[..., ::-1]  # Convert BGR to RGB
        ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)  # Read annotation as grayscale

        if Img is None or ann_map is None:
            print(f"Error: Could not read image or mask from path {ent['image']} or {ent['annotation']}")
            continue

        # Resize image and mask
        target_size = (1024, 1024)
        # r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # Scaling factor
        Img = cv2.resize(Img, target_size)
        ann_map = cv2.resize(ann_map, target_size, interpolation=cv2.INTER_NEAREST)

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

        # Get all coordinates inside the eroded mask and choose random points
        coords = np.argwhere(eroded_mask > 0)
        if len(coords) > 0:
            for _ in inds:  # Select as many points as there are unique labels
                yx = np.array(coords[np.random.randint(len(coords))])
                points.append([yx[1], yx[0]])

        points = np.array(points)

        # Prepare binary mask for return
        binary_mask = np.expand_dims(binary_mask, axis=-1)  # Now shape is (1024, 1024, 1)
        binary_mask = binary_mask.transpose((2, 0, 1))  # Shape: (1, 1024, 1024)
        points = np.expand_dims(points, axis=1)  # Shape: (num_points, 1, 2)

        if visualize_data:
            # Plotting the images and points
            plt.figure(figsize=(15, 5))
            # Original Image
            plt.subplot(1, 3, 1)
            plt.title('Original Image')
            plt.imshow(Img)
            plt.axis('off')
            # Segmentation Mask (binary_mask)
            plt.subplot(1, 3, 2)
            plt.title('Binarized Mask')
            plt.imshow(binary_mask, cmap='gray')
            plt.axis('off')
            # Mask with Points in Different Colors
            plt.subplot(1, 3, 3)
            plt.title('Binarized Mask with Points')
            plt.imshow(binary_mask, cmap='gray')
            # Plot points in different colors
            colors = list(mcolors.TABLEAU_COLORS.values())
            for i, point in enumerate(points):
                plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100, label=f'Point {i+1}')  # Corrected to plot y, x order
            # plt.legend()
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        # Store results for this entry
        all_images.append(Img)
        all_masks.append(binary_mask)
        all_points.append(points)

    # Return lists of images, masks, points, and number of masks for each entry
    all_images = np.array(all_images)
    all_masks = np.array(all_masks)
    all_points = np.array(all_points)
    return all_images, all_masks, all_points

class SAMDataset(Dataset):
    """
    This class is used to create a dataset that serves input images and masks.
    It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
    """
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = np.array(item["image"])
        mask = np.array(item["label"])
        prompt = item['prompt']

        # prepare image and prompt for the model
        inputs = self.processor(image, input_points=[prompt], return_tensors="pt")
        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        # add ground truth segmentation
        inputs["ground_truth_mask"] = mask
        return inputs

def train(num_image, batch_size, num_epoch):

    # 対象のディレクトリを指定
    mask_dir = 'basketball-instants-dataset//train_mask'
    image_dir = 'basketball-instants-dataset//train_image'
    # ディレクトリ内のファイルを順に参照するループ
    count = 0
    images = []
    masks = []
    prompts = []
    for mask_filename in os.listdir(mask_dir):
        if count == num_image:
            break
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
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    # Create an instance of the SAMDataset
    train_dataset = SAMDataset(dataset=dataset, processor=processor)
    example = train_dataset[0]
    '''for k,v in example.items():
        print(k,v.shape)'''
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    # 直接バッチを取得（イテレータにしない）
    for batch in train_dataloader:
        # バッチ内の各要素の形状を確認
        for k, v in batch.items():
            print(k, v.shape)
        break  # ここで1バッチだけ確認したい場合はbreakでループを終了'''
    # load the model
    ########################### ここ変えてしまった　########################### 
    sam2_checkpoint = "sam2_hiera_base_plus.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
    model_cfg = "sam2_hiera_b+.yaml" # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    # Train mask decoder.
    predictor.model.sam_mask_decoder.train(True)
    # Train prompt encoder.
    predictor.model.sam_prompt_encoder.train(True)
    # Initialize the optimizer and the loss function
    # optimizer = Adam(sam2_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=0.0001,weight_decay=1e-4) #1e-5, weight_decay = 4e-5
    ########################### ここ変えてしまった　########################### 
    #Try DiceFocalLoss, FocalLoss, DiceCELoss
    # seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    # Mix precision.
    scaler = torch.cuda.amp.GradScaler()
    #Try DiceFocalLoss, FocalLoss, DiceCELoss
    # seg_loss = dice_loss()
    #Training loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model.to(device)
    sam2_model.train()

    # Fine-tuned model name.
    FINE_TUNED_MODEL_NAME = "fine_tuned_sam2"

    print('--start training--')
    best_loss = float('inf')
    best_epoch = 0
    all_epoch_losses = []
    for epoch in range(num_epoch):
        epoch_losses = []
        for batch_ind, batch_data in enumerate(train_dataloader):
            # forward pass
            outputs = sam2_model(pixel_values=batch_data["pixel_values"].to(device),
                            input_prompts=batch_data["input_points"].to(device),
                            multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch_data["ground_truth_mask"].float().to(device)
            loss = dice_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch
                # 最良のIoUを持つモデルを保存
                FINE_TUNED_MODEL = FINE_TUNED_MODEL_NAME + "_best.torch"
                save_path = "fine_tuned_model//" + str(num_image) + "//" + FINE_TUNED_MODEL
                torch.save(sam2_model.state_dict(), save_path)
                print(f"New best model saved at step {best_epoch} with IoU {best_loss}")

        mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        all_epoch_losses.append(mean_epoch_loss)
        print(f'EPOCH: {epoch}, Mean loss: {mean_epoch_loss:.6f}')

    # Plot and save the loss graph
    plt.figure()
    plt.plot(range(num_epoch), all_epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('visualize_training//' + str(num_image) + '//training_loss_plot.png')

if __name__ == '__main__':
    args = parse_arguments()
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    num_images = [int(num_image) for num_image in args.num_image.split(",")]
    for num_image in num_images:
        train(num_image, batch_size, num_epoch)

    '''best_iou = 0
    best_step = 0
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2) # 500 , 250, gamma = 0.1
    accumulation_steps = 4  # Number of steps to accumulate gradients before updating
    image, mask, input_point= read_batch(train_data, visualize_data=False)
    print('--finish read batch--')
    for step in range(1, NO_OF_STEPS + 1):
        with torch.cuda.amp.autocast():
            if image is None or mask is None:
                continue

            input_label = np.ones((num_image, 1))
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                continue

            if input_point.size == 0 or input_label.size == 0:
                continue

            # 例えば (batch_size, height, width, channels) の場合
            for i in range(image.shape[0]):
                single_image = image[i]  # バッチ内の1枚の画像を取得

                predictor.set_image(image)
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
                if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                    continue

                # プロンプト埋め込みの生成
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels), boxes=None, masks=None,
                )

                batched_mode = unnorm_coords.shape[0] > 1
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                
                print('--start prediction--')
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )
                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

                gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
                prd_mask = torch.sigmoid(prd_masks[:, 0])
                seg_loss = (-gt_mask * torch.log(prd_mask + 0.000001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

                inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                loss = seg_loss + score_loss * 0.05

                # Apply gradient accumulation
                loss = loss / accumulation_steps
                scaler.scale(loss).backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

            if step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                predictor.model.zero_grad()

            # Update scheduler
            scheduler.step()

            if step % 500 == 0:
                FINE_TUNED_MODEL = FINE_TUNED_MODEL_NAME + "_" + str(step)+ ".torch"
                save_path = "fine_tuned_model//" + str(num_image) + "//" + FINE_TUNED_MODEL
                torch.save(predictor.model.state_dict(), save_path)

            if step == 1:
                mean_iou = 0

            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

            # 最良のIoUを更新し、そのステップでモデルを保存
            current_mean_iou = np.mean(iou.cpu().detach().numpy())
            if current_mean_iou > best_iou:
                best_iou = current_mean_iou
                best_step = step
                # 最良のIoUを持つモデルを保存
                FINE_TUNED_MODEL = FINE_TUNED_MODEL_NAME + "_best.torch"
                save_path = "fine_tuned_model//" + str(num_image) + "//" + FINE_TUNED_MODEL
                torch.save(predictor.model.state_dict(), save_path)
                print(f"New best model saved at step {best_step} with IoU {best_iou}")

            if step % 100 == 0:
                print("Step " + str(step) + ":\t", "Accuracy (IoU) = ", mean_iou)
'''