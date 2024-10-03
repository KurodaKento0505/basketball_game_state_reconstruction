import json
import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
from ultralytics import SAM
from pycocotools import mask as mask_utils  # RLEデコード用
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase')
    return parser.parse_args()

def show_bbox(image, bbox_per_image_list):
    """
    画像に複数のバウンディングボックスを描画する関数

    Parameters:
    image (numpy.ndarray): cv2.imread(image_path)で読み取られた画像
    bbox_per_image_list (list): バウンディングボックスのリスト。各要素は左上のx座標、左上のy座標、横の長さ、縦の長さの順に値を持つ。

    Returns:
    重ねた画像
    """
    # バウンディングボックスの色と太さを設定
    color = (0, 255, 0)  # 緑色
    thickness = 2
    # 各バウンディングボックスを描画
    for bbox in bbox_per_image_list:
        x, y, w, h = map(int, bbox)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image 

def make_bbox_file(phase):

    # JSONファイルの読み込み
    with open("annotations/" + phase + ".json", "r") as f:
        annotation_data = json.load(f)
        images_list = annotation_data["images"]
        annotation_list = annotation_data["annotations"]

    num_image = 0
    bbox_dict = {}

    for image in images_list:
        image_dict = image

        # 対応する annotation を見つける
        annotations_per_image_list = []
        for annotation in annotation_list:
            if annotation['image_id'] == image_dict['id']:
                annotations_per_image_list.append(annotation)
            if int(annotation['image_id']) > int(image_dict['id']):
                break  # 対応する image が見つかったらループを終了
        if annotations_per_image_list is None:
            print(f"Image id {image_dict['id']} に対応する image が見つかりませんでした")
            continue

        # 画像の読み込み
        image = cv2.imread(os.path.join("basketball-instants-dataset", image_dict["file_name"]))
        image_path = os.path.join("dataset_yolo/" + phase + "_image", f"{num_image}.png")
        cv2.imwrite(image_path, image)

        # JSONアノテーションのセグメンテーションをデコード
        bbox_per_image_list = []
        for annotation_dict in annotations_per_image_list:
            bbox_per_image_list.append(annotation_dict['bbox'])

        bbox_dict[num_image] = bbox_per_image_list
        
        # 可視化
        # bbox_image = show_bbox(image, bbox_per_image_list)
        # 保存
        '''bbox_image_path = os.path.join("dataset_yolo/bbox_image", f"{annotation_dict['id']}.png")
        cv2.imwrite(bbox_image_path, bbox_image)
        print('bbox_image_path:', bbox_image_path)'''

        num_image += 1

    with open('dataset_yolo/' + phase + '_bbox/bbox.json', 'w') as f:
        json.dump(bbox_dict, f, indent=2)

if __name__ == '__main__':
    args = parse_arguments()
    phase = args.phase
    make_bbox_file(phase)