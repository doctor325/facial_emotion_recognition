"""
快速检查CSV数据分布
"""
import pandas as pd

# 读取训练数据
df = pd.read_csv('data/train.csv')

print("="*60)
print("FER2013 训练集数据分布")
print("="*60)

emotion_labels = {
    0: 'anger (愤怒)',
    1: 'disgust (厌恶)',
    2: 'fear (恐惧)',
    3: 'happy (高兴)',
    4: 'sad (悲伤)',
    5: 'surprise (惊讶)',
    6: 'neutral (中性)'
}

print(f"\n总样本数: {len(df)}")
print("\n各类别分布:")
print("-" * 60)

for emotion in range(7):
    count = len(df[df['emotion'] == emotion])
    percentage = count / len(df) * 100
    print(f"{emotion} - {emotion_labels[emotion]:15s}: {count:5d} 张 ({percentage:5.2f}%)")

print("-" * 60)
print("\n数据分布是否均衡:", "是" if df['emotion'].value_counts().max() / df['emotion'].value_counts().min() < 2 else "否（存在类别不平衡）")
