"""
将FER2013 CSV数据集转换为ImageFolder格式
数据目录结构:
data/
  train/
    0/ (anger)
    1/ (disgust)
    2/ (fear)
    3/ (happy)
    4/ (sad)
    5/ (surprise)
    6/ (neutral)
  val/
    0-6/
  test/
    0-6/
"""

import pandas as pd
import numpy as np
from PIL import Image
import os

def create_image_folder(csv_file, output_dir, split_ratios={'train': 0.8, 'val': 0.1, 'test': 0.1}):
    """
    将CSV文件转换为ImageFolder格式
    
    参数:
    csv_file: CSV文件路径
    output_dir: 输出目录
    split_ratios: 数据集划分比例
    """
    
    # 读取CSV文件
    print(f"读取CSV文件: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"总共 {len(df)} 张图像")
    print(f"情感类别: {df['emotion'].unique()}")
    print(f"各类别样本数量:")
    print(df['emotion'].value_counts().sort_index())
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 情感标签映射
    emotion_labels = {
        0: 'anger',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    
    # 按类别分组
    for emotion in range(7):
        emotion_data = df[df['emotion'] == emotion]
        print(f"\n处理情感类别: {emotion} ({emotion_labels[emotion]})")
        print(f"样本数量: {len(emotion_data)}")
        
        # 随机打乱数据
        emotion_data = emotion_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 划分数据集
        n_total = len(emotion_data)
        n_train = int(n_total * split_ratios['train'])
        n_val = int(n_total * split_ratios['val'])
        
        splits = {
            'train': emotion_data[:n_train],
            'val': emotion_data[n_train:n_train+n_val],
            'test': emotion_data[n_train+n_val:]
        }
        
        # 保存图像
        for split_name, split_data in splits.items():
            split_dir = os.path.join(output_dir, split_name, str(emotion))
            os.makedirs(split_dir, exist_ok=True)
            
            for idx, row in split_data.iterrows():
                # 解析像素数据
                pixels = np.array(list(map(int, row['pixels'].split())))
                
                # 重塑为48x48图像
                image_array = pixels.reshape(48, 48).astype(np.uint8)
                
                # 创建PIL图像
                image = Image.fromarray(image_array, mode='L')
                
                # 保存图像
                image_path = os.path.join(split_dir, f'{idx}_{emotion}.png')
                image.save(image_path)
            
            print(f"  {split_name}: {len(split_data)} 张图像 -> {split_dir}")
    
    print("\n" + "="*50)
    print("数据转换完成!")
    print("="*50)
    print("\n数据目录结构:")
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            total_images = sum([len(os.listdir(os.path.join(split_dir, str(e)))) 
                               for e in range(7) 
                               if os.path.exists(os.path.join(split_dir, str(e)))])
            print(f"  {split}/: {total_images} 张图像")
            for emotion in range(7):
                emotion_dir = os.path.join(split_dir, str(emotion))
                if os.path.exists(emotion_dir):
                    n_images = len(os.listdir(emotion_dir))
                    print(f"    {emotion}/: {n_images} 张图像")

def main():
    # 配置 - 使用绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CSV_FILE = os.path.join(script_dir, 'data', 'train.csv')  # 输入CSV文件
    OUTPUT_DIR = os.path.join(script_dir, 'data')  # 输出目录
    
    # 检查CSV文件是否存在
    if not os.path.exists(CSV_FILE):
        print(f"错误: 未找到CSV文件 {CSV_FILE}")
        print("请确保CSV文件存在于 data/ 目录中")
        return
    
    print(f"CSV文件: {CSV_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*50)
    
    # 转换数据
    create_image_folder(CSV_FILE, OUTPUT_DIR)

if __name__ == "__main__":
    main()
