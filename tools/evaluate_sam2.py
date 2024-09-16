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

# RLEマスクをデコードし、マスクの中心点を返す関数
def decode_rle(rle_segmentation):
    # RLE形式をデコード
    rle = {
        "size": rle_segmentation["size"],
        "counts": rle_segmentation["counts"].encode()  # countsはバイナリデータである必要がある
    }
    decoded_masks = mask_utils.decode(rle)
    # マスクに含まれるオブジェクトをラベリング
    labeled_mask, num_objects = label(decoded_masks)

    # マスクの中で値が1のピクセル（オブジェクトが存在する部分）の座標を取得
    # 各オブジェクトの重心を計算
    centroids = []
    for obj_label in range(1, num_objects + 1):
        # 各ラベル（オブジェクト）に対して重心を計算
        centroid = center_of_mass(decoded_masks, labeled_mask, obj_label)
        # (row, col)の順で返ってくるため、(x, y)に変換し整数に
        centroids.append((int(centroid[1]), int(centroid[0])))
    centroids = np.array(centroids)
    return decoded_masks, centroids, num_objects

def mask_annotation(image, decoded_mask):
    # マスクを0~1の範囲に正規化（255で割る）
    mask_normalized = decoded_mask.astype(np.float32)
    # 元画像とマスクを重ねる
    alpha = 0.5  # マスクの透明度（0=完全に透明、1=完全に不透明）
    overlay_color = np.array([0, 0, 255])  # マスクの色（ここでは赤）
    # マスク部分だけに色をつける（元画像に半透明の赤を重ねる）
    mask_colored = np.zeros_like(image, dtype=np.float32)
    for i in range(3):  # RGB各チャンネルにマスクの色を適用
        mask_colored[:, :, i] = mask_normalized * overlay_color[i]
    # 透明度を適用して元画像とマスクを合成
    overlay_image = image.astype(np.float32)
    cv2.addWeighted(mask_colored, alpha, overlay_image, 1 - alpha, 0, overlay_image)
    cv2.imshow("Original Image with Mask Overlay", overlay_image.astype(np.uint8))

# 点群の表示
def show_points(coords, labels, ax, marker_size=375):
    # labelsがリストや1次元配列であることを前提に
    labels = np.array(labels)  # labelsをnumpy配列に変換
    # labelsが1次元であることを確認する
    if labels.ndim == 1:
        pos_points = coords[0]
        # プロット
        ax.scatter(pos_points[0], pos_points[1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    else:
        raise ValueError("labels must be a 1D array of integers representing the point labels (1 for foreground, 0 for background).")

# プロンプト表示のための関数
def show_prompt(image, input_point, input_label):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    # 入力が2次元座標と1次元のラベルであることを確認する
    # input_point = np.array(input_point)  # input_pointをnumpy配列に変換
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()   

# マスクの表示
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

# マスク群の表示
def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

# 閾値以下の面積のマスクを排除
def remove_small_mask(masks, scores):
    # マスク面積の閾値を設定（例: 1000ピクセル以上のマスクのみ残す）
    threshold_area = 10000

    # 面積が閾値以上のマスクをフィルタリング
    filtered_masks = []
    filtered_scores = []

    for i, mask in enumerate(masks):
        # マスクの面積を計算
        mask_area = np.sum(mask)

        # 閾値以上の面積があるマスクのみを残す
        if mask_area > threshold_area:
            filtered_masks.append(mask)
            filtered_scores.append(scores[i])

    # フィルタリングされたマスクとスコアをNumPy配列に変換
    filtered_masks = np.array(filtered_masks)
    filtered_scores = np.array(filtered_scores)

    return filtered_masks, filtered_scores

# SAM2とアノテーションの比較（IoUを計算）
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

if __name__ == '__main__':

    # JSONファイルの読み込み
    with open("annotations\\test.json", "r") as f:
        annotation_data = json.load(f)
        images_list = annotation_data["images"]
        annotation_list = annotation_data["annotations"]

    # Load a model
    checkpoint = "checkpoints//sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device="cuda"))

    # model = SAM("sam2_b.pt")

    # 結果を格納するリスト
    all_results_list = []

    for i in range(len(annotation_list)):

        annotation_dict = annotation_list[i]
        # 対応するアノテーションを探す
        image_dict = None
        for image in images_list:
            if image['id'] == annotation_dict['image_id']:
                image_dict = image
                break  # 対応するアノテーションが見つかったらループを終了

        if image_dict is None:
            print(f"Image id {annotation_dict['image_id']} に対応するアノテーションが見つかりませんでした")
            continue

        # 画像の読み込み
        image_path = os.path.join("basketball-instants-dataset", image_dict["file_name"])
        image = cv2.imread(image_path)

        # JSONアノテーションのセグメンテーションをデコード
        decoded_masks, centroids, num_objects = decode_rle(annotation_dict['segmentation'])
        # 元画像に正解データのマスクを重ねる
        # mask_annotation(image, decoded_mask)

        # SAM2でのセグメンテーション
        sam_segmentation = predictor.set_image(image)

        input_point = centroids  # 位置
        input_label = np.array([1] * num_objects)  # ラベル (1:前景点、0:背景点)

        # 元画像にプロンプトを重ねる
        # show_prompt(image, input_point, input_label)

        # Predictorで予測
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
            )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind] # マスク
        scores = scores[sorted_ind] # スコア
        logits = logits[sorted_ind] # ロジット

        # 最もIoUのスコアが高かったマスクを記録
        best_IoU = -1
        for k in range(len(masks)):
            iou_score = calculate_iou(decoded_masks, masks[k])
            if iou_score > best_IoU:
                best_IoU = iou_score
                best_mask = masks[k]
        # print(f"SAM2とアノテーションのIoUスコア: {best_IoU:.4f}")

        all_results_list.append(best_IoU)

        '''################# 描画 #################
        annotated_image = image.copy()
        annotated_image[decoded_masks == 1] = [0, 255, 0]  # 緑色で正解マスクを表示
        # 予測結果でbest_IoUのマスクをすべて重ねて描画
        predicted_image = image.copy()
        predicted_image[best_mask == 1] = [0, 0, 255]  # 赤色で予測マスクを表示
        # 正解マスクと予測マスクを1枚に重ねる
        combined_image = cv2.addWeighted(annotated_image, 0.5, predicted_image, 0.5, 0)
        # 結果の表示
        cv2.imshow(f"Image {i}: Annotation (Green) vs Prediction (Red)", combined_image)
        cv2.waitKey(0)  # キーが押されるまで待機'''

        if i % 100 == 0:
            print(i)

    print('mean_all_results_list:', sum(all_results_list) / len(all_results_list))