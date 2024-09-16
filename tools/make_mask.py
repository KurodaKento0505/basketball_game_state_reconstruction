import json
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
from ultralytics import SAM
from pycocotools import mask as mask_utils  # RLEデコード用
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from evaluate_sam2 import decode_rle

if __name__ == '__main__':

    # JSONファイルの読み込み
    with open("annotations\\test.json", "r") as f:
        annotation_data = json.load(f)
        images_list = annotation_data["images"]
        annotation_list = annotation_data["annotations"]

    for i in range(len(annotation_list)):

        # 対応する image を見つける
        annotation_dict = annotation_list[i]
        image_dict = None
        for image in images_list:
            if image['id'] == annotation_dict['image_id']:
                image_dict = image
                break  # 対応する image が見つかったらループを終了
        if image_dict is None:
            print(f"Image id {annotation_dict['image_id']} に対応する image が見つかりませんでした")
            continue

        # 画像の読み込み
        image_path = os.path.join("basketball-instants-dataset", image_dict["file_name"])
        image = cv2.imread(image_path)

        # JSONアノテーションのセグメンテーションをデコード
        decoded_masks, centroids, num_objects = decode_rle(annotation_dict['segmentation'])
        
        # path
        mask_path = os.path.join("basketball-instants-dataset\\test_mask", f"{annotation_dict['id']}.png")
        image_path = os.path.join("basketball-instants-dataset\\test_image", f"{annotation_dict['id']}.png")

        cv2.imwrite(mask_path, decoded_masks * 255)
        cv2.imwrite(image_path, image)