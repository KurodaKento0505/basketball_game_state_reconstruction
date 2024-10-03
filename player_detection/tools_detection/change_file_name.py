import os

# ファイルが保存されているディレクトリのパス
dir_path = '/root/basketball/player_detection/dataset_yolo/10/valid/labels'

# ディレクトリ内の全てのファイル名を取得
files = os.listdir(dir_path)

# PNGファイルに絞る
png_files = [f for f in files if f.endswith('.txt')]

# ファイル名をゼロ埋めで変換
for file in png_files:
    # 拡張子を除いたファイル名を取得
    base_name = os.path.splitext(file)[0]
    
    # 数字の部分を3桁にゼロ埋めする
    new_name = f"{int(base_name):03}.txt"
    
    # 元ファイルと新しいファイル名のフルパス
    old_file = os.path.join(dir_path, file)
    new_file = os.path.join(dir_path, new_name)
    
    # ファイル名を変更
    os.rename(old_file, new_file)
    print(f"Renamed {file} to {new_name}")
