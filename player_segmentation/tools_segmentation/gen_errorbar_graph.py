import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_file')
    return parser.parse_args()

def gen_graph(num_file):
    # 三つのJSONファイルのデータを用意します（仮にデータが入った変数として）
    # データを格納するリスト
    all_data = []
    # ファイルの読み込み
    for i in range(1, num_file + 1):
        file_path = f"visualize_inference/all/iou_results_{i}.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_data.append(data)
    # num_imageごとのmean_iouを格納する辞書
    num_images = [entry['num_image'] for entry in all_data[0]]  # num_imageはすべてのファイルで同じ
    mean_ious = {num_image: [] for num_image in num_images}

    # num_imageごとにmean_iouをまとめる
    for data in all_data:
        for entry in data:
            mean_ious[entry['num_image']].append(entry['mean_iou'])

    # 平均と標準偏差を計算
    mean_values = [np.mean(mean_ious[num_image]) for num_image in num_images]
    std_values = [np.std(mean_ious[num_image]) for num_image in num_images]

    # 等間隔にするためのx軸の位置を設定
    x_positions = np.arange(len(num_images))  # 等間隔のx位置

    # グラフを描画
    plt.figure(figsize=(8, 6))
    plt.errorbar(x_positions, mean_values, yerr=std_values, fmt='-o', capsize=5, label="Mean IoU with Error Bars")
    # 横軸の目盛りを等間隔に設定し、ラベルを元のnum_imagesに
    plt.xticks(x_positions, num_images)  # 等間隔のx位置に元のnum_imagesのラベルを対応させる
    # 標準偏差を平均値の左上に表示
    for x, mean, std in zip(x_positions, mean_values, std_values):
        plt.text(x - 0.2, mean + 0.01, f"{std:.2f}", fontsize=10, color='red')
    plt.xlabel("Number of Images")
    plt.ylabel("Mean IoU")
    plt.grid(True)
    plt.legend()
    plt.savefig("visualize_inference//all//change_epoch.png")
    plt.show()

if __name__ == '__main__':
    args = parse_arguments()
    num_file = args.num_file
    gen_graph(int(num_file))