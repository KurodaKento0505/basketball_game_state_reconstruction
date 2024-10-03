import os
import cv2
import json
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase')
    return parser.parse_args()

# 画像の幅と高さの取得
def get_image_shape(image):
    image_height, image_width = image.shape[:2]
    return image_height, image_width

# ラベルデータの変換関数
def convert_to_yolo_format(label_dict, image_width, image_height, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for image_id, bboxes in label_dict.items():
        label_file = os.path.join(output_dir, f"{image_id}.txt")
        with open(label_file, "w") as f:
            for bbox in bboxes:
                x, y, w, h = bbox
                # 中心座標を計算
                center_x = x + w / 2
                center_y = y + h / 2
                
                # 相対座標に正規化
                center_x /= image_width
                center_y /= image_height
                w /= image_width
                h /= image_height
                
                # class_id (0と仮定)
                class_id = 0
                
                # YOLO形式のデータを書き込み
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {w:.6f} {h:.6f}\n")

def convert(phase):
    # テスト用データ
    json_file_path = 'dataset_yolo/' + phase + '_bbox/bbox.json'
    with open(json_file_path, "r") as f:
        label_dict = json.load(f)

    # 画像の幅と高さを設定
    image_path = 'dataset_yolo/' + phase + '_image/0.png'
    image = cv2.imread(image_path)
    image_height, image_width = get_image_shape(image)

    # 出力先のディレクトリ
    output_dir = 'dataset_yolo/' + phase + '_label'

    # ラベルデータを変換して保存
    convert_to_yolo_format(label_dict, image_width, image_height, output_dir)

if __name__ == '__main__':
    args = parse_arguments()
    phase = args.phase
    convert(phase)
