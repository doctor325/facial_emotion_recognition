"""
检查数据集分布和验证数据正确性
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 修复中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def check_data_distribution(data_dir='data'):
    """检查数据分布"""
    emotion_labels = {
        0: 'anger',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    
    print("="*60)
    print("数据集分布检查")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"\n警告: {split} 目录不存在")
            continue
        
        print(f"\n{split.upper()} 集:")
        print("-" * 60)
        
        total_images = 0
        for emotion in range(7):
            emotion_dir = os.path.join(split_dir, str(emotion))
            if os.path.exists(emotion_dir):
                n_images = len(os.listdir(emotion_dir))
                total_images += n_images
                print(f"  {emotion} ({emotion_labels[emotion]:8s}): {n_images:5d} 张")
        
        print(f"  {'总计':8s}: {total_images:5d} 张")
    
    print("\n" + "="*60)

def visualize_sample_images(data_dir='data', num_samples=5):
    """可视化每个类别的样本图像"""
    emotion_labels = {
        0: 'anger',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    
    fig, axes = plt.subplots(7, num_samples, figsize=(15, 10))
    
    for emotion in range(7):
        emotion_dir = os.path.join(data_dir, 'train', str(emotion))
        if not os.path.exists(emotion_dir):
            continue
        
        image_files = os.listdir(emotion_dir)[:num_samples]
        
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(emotion_dir, img_file)
            img = Image.open(img_path)
            img_array = np.array(img)
            
            axes[emotion, i].imshow(img_array, cmap='gray')
            axes[emotion, i].set_title(f'{emotion_labels[emotion]}', fontsize=10)
            axes[emotion, i].axis('off')
    
    plt.suptitle('各类别人脸情感样本示例', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=200, bbox_inches='tight')
    print("\n样本图像已保存为 sample_images.png")
    plt.show()

if __name__ == "__main__":
    # 检查数据分布
    check_data_distribution()
    
    # 可视化样本
    print("\n正在生成样本图像示例...")
    visualize_sample_images()
