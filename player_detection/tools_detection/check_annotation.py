import cv2
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_image')
    return parser.parse_args()

# YOLOラベルを読み込む関数
def load_yolo_labels(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            # クラスID, x_center, y_center, width, height を読み込む
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            boxes.append([class_id, x_center, y_center, width, height])
    return boxes

# バウンディングボックスを描画する関数
def draw_boxes(image, boxes, img_width, img_height):
    for box in boxes:
        class_id, x_center, y_center, width, height = box
        # YOLO形式の座標（正規化された値）をピクセル単位に変換
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # バウンディングボックスの左上と右下の座標を計算
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # バウンディングボックスを描画
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # クラスIDを表示
        cv2.putText(image, f'Class {int(class_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 画像とラベルファイルをループで処理
def check_annotation(image_dir, label_dir, output_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):  # 画像ファイルのみを対象
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.png', '.txt'))

            # 対応するラベルファイルが存在するか確認
            if os.path.exists(label_path):
                # 画像の読み込み
                image = cv2.imread(image_path)
                img_height, img_width = image.shape[:2]

                # ラベルを読み込んでバウンディングボックスを取得
                boxes = load_yolo_labels(label_path)

                # 画像にバウンディングボックスを描画
                draw_boxes(image, boxes, img_width, img_height)

                # 結果を保存
                output_image_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_image_path, image)
                print(f'Saved: {output_image_path}')
            else:
                print(f'Label file not found for: {image_path}')

if __name__ == '__main__':
    args = parse_arguments()
    num_images = [int(num_image) for num_image in args.num_image.split(",")]
    for num_image in num_images:
        # 画像とラベルのディレクトリパス
        image_dir = '/root/basketball/player_detection/dataset_yolo/' + str(num_image) + '/valid/images'
        label_dir = '/root/basketball/player_detection/dataset_yolo/' + str(num_image) + '/valid/labels'
        output_dir = '/root/basketball/player_detection/dataset_yolo/' + str(num_image) + '/train/output_images'

        # 出力ディレクトリを作成（存在しない場合）
        os.makedirs(output_dir, exist_ok=True)

        check_annotation(image_dir, label_dir, output_dir)