
import os
import pandas as pd

def extract_image_ids(folder_path, csv_path):
    image_ids = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_id = os.path.splitext(filename)[0]
            image_ids.append(image_id)

    df = pd.read_csv(csv_path)
    df['id'] = image_ids
    df.to_csv(csv_path, index=False)

# 示例用法
folder_path = '/path/to/your/folder'  # 替换为实际的文件夹路径
csv_path = '/path/to/your/csv/file.csv'  # 替换为实际的CSV文件路径

extract_image_ids(folder_path, csv_path)
