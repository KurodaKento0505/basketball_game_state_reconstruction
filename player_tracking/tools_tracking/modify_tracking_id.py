import json
import os
import cv2
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_video')
    return parser.parse_args()

def modify_tracking_results(num_video):
    """
    YOLOの検出結果から、1フレーム目に出てきたIDのみを維持し、
    それ以降のフレームで新しいIDを排除し、消えたIDを補完する処理を行い、結果を新しいJSONファイルに保存します。
    """
    
    # 元のJSONファイルを読み込む
    DETECTION_RESULTS_PATH = f'player_tracking/predict_frame_yolo/{num_video}.json'
    MODIFIED_RESULTS_PATH = f'player_tracking/predict_frame_yolo/{num_video}_modified.json'
    os.makedirs(os.path.dirname(MODIFIED_RESULTS_PATH), exist_ok=True)

    with open(DETECTION_RESULTS_PATH, 'r') as json_file:
        all_detections = json.load(json_file)
    
    # 1フレーム目のIDを記憶
    first_frame_detections = all_detections[0]['detections']
    first_frame_ids = {detection['id'] for detection in first_frame_detections}

    # 新しい検出結果を格納するリスト
    modified_detections = []

    # 各フレームで処理を行う
    for frame_data in all_detections:
        frame_num = frame_data['frame']
        current_ids = {detection['id'] for detection in frame_data['detections']}
        
        # 1フレーム目にいたが、現在いなくなったIDを補完
        missing_ids = list(first_frame_ids - current_ids)
        # 新しく現れたIDを排除
        extra_ids = list(current_ids - first_frame_ids)

        # IDを置き換えた検出結果を格納するリスト
        new_detections = []
        extra_index = 0  # extra_idsのインデックス
        
        for detection in frame_data['detections']:
            new_detection = detection.copy()  # 検出結果をコピー

            if detection['id'] in extra_ids and extra_index < len(missing_ids):
                # 新しいIDが見つかった場合、欠けたIDで置き換える
                new_detection['id'] = missing_ids[extra_index]
                extra_index += 1

            new_detections.append(new_detection)
        
        # 修正した検出結果を新しいリストに追加
        modified_detections.append({
            "frame": frame_num,
            "detections": new_detections
        })

    # 新しいJSONファイルに保存
    with open(MODIFIED_RESULTS_PATH, 'w') as modified_file:
        json.dump(modified_detections, modified_file, indent=4)

def draw_bboxes_on_video(num_video):
    """
    YOLOで検出したバウンディングボックスとIDを動画上に描画し、新しい動画ファイルを出力する。
    """
    # JSONファイルの読み込み
    DETECTION_RESULTS_PATH = f'player_tracking/predict_frame_yolo/{num_video}_modified.json'
    with open(DETECTION_RESULTS_PATH, 'r') as json_file:
        all_detections = json.load(json_file)

    # 入力動画と出力動画のパス
    SOURCE_VIDEO_PATH = f'basketball-video-dataset/video/{num_video}.mp4'
    OUTPUT_VIDEO_PATH = f'player_tracking/predict_video/track_modified_{num_video}.mp4'
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

    # 動画の読み込み
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    frame_count = 0

    # 各フレームを順に処理
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 該当フレームの検出結果を取得
        frame_detections = next((item for item in all_detections if item["frame"] == frame_count), None)
        if frame_detections:
            for detection in frame_detections["detections"]:
                # バウンディングボックスを描画
                box = detection["box"]
                x_min, y_min = int(box["x_min"]), int(box["y_min"])
                x_max, y_max = int(box["x_max"]), int(box["y_max"])
                id = detection["id"]

                # バウンディングボックスを描く
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # IDを描画
                cv2.putText(frame, f'ID: {id}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # フレームを書き出す
        out.write(frame)

        # 次のフレームへ
        frame_count += 1

    # リソースの解放
    cap.release()
    out.release()

if __name__ == '__main__':
    args = parse_arguments
    # num_video = args.num_video
    num_video = str(5)
    modify_tracking_results(num_video)
    draw_bboxes_on_video(num_video)