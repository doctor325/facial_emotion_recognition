# 人脸情感识别系统

基于CNN的人脸情感识别系统，使用FER2013数据集训练。

## 项目结构

```
facial_emotion_recognition/
├── model.py              # 模型定义和训练函数
├── train.py              # 训练脚本
├── app.py                # Streamlit Web应用
├── convert_data.py       # 数据转换脚本
├── data/                 # 数据目录
│   ├── train.csv         # 训练数据(CSV格式)
│   ├── train/            # 训练集(ImageFolder格式)
│   ├── val/              # 验证集
│   └── test/             # 测试集
└── best_model.pth        # 训练好的模型权重
```

## 环境要求

```
torch>=1.7.0
torchvision>=0.8.0
streamlit>=1.0.0
matplotlib>=3.3.0
pandas>=1.0.0
numpy>=1.19.0
Pillow>=7.0.0
```

## 安装依赖

```bash
pip install torch torchvision streamlit matplotlib pandas numpy Pillow
```

## 使用步骤

### 1. 准备数据集

#### 选项A：使用已有的FER2013 CSV文件

如果您已经有FER2013的CSV文件（如 `data/train.csv`），运行以下命令转换为ImageFolder格式：

```bash
python convert_data.py
```

这将会自动将数据划分为训练集(80%)、验证集(10%)和测试集(10%)。

#### 选项B：手动准备数据

按照以下目录结构组织数据：

```
data/
├── train/
│   ├── 0/  (anger)
│   ├── 1/  (disgust)
│   ├── 2/  (fear)
│   ├── 3/  (happy)
│   ├── 4/  (sad)
│   ├── 5/  (surprise)
│   └── 6/  (neutral)
├── val/
│   └── 0-6/
└── test/
    └── 0-6/
```

每个文件夹中放置对应的48x48灰度PNG图像。

### 2. 训练模型

```bash
python train.py
```

训练过程会：
- 自动加载数据
- 训练CNN模型（最多50个epoch）
- 使用早停机制防止过拟合（patience=10）
- 保存最佳模型为 `best_model.pth`
- 生成训练曲线图 `training_history.png`
- 在测试集上评估最终性能

#### 训练参数调整

如需调整训练参数，编辑 `train.py` 中的配置：

```python
DATA_DIR = 'data'           # 数据目录
BATCH_SIZE = 64             # 批次大小
NUM_EPOCHS = 50             # 最大训练轮数
LEARNING_RATE = 0.001       # 学习率
```

### 3. 运行Web应用

```bash
streamlit run app.py
```

应用会自动在浏览器中打开，您可以：
- 上传人脸图像
- 点击"识别情感"按钮
- 查看预测结果和置信度
- 查看各类别概率分布图

## 模型架构

CNN模型结构：
- 3个卷积层（32, 64, 128个滤波器）
- 最大池化层
- Dropout正则化（0.5）
- 2个全连接层（512, 7个神经元）
- ReLU激活函数

## 情感类别

模型识别7种情感：
- 0: anger（生气）
- 1: disgust（厌恶）
- 2: fear（恐惧）
- 3: happy（开心）
- 4: sad（伤心）
- 5: surprise（惊讶）
- 6: neutral（中性）

## 数据集说明

FER2013数据集：
- 图像大小：48x48像素，灰度图
- 训练样本：约28,000张
- 测试样本：约3,500张
- 7种情感类别

## 后续改进建议

1. **数据增强**
   - 添加随机旋转、翻转、平移
   - 调整亮度和对比度

2. **模型改进**
   - 尝试ResNet、EfficientNet等预训练模型
   - 使用迁移学习
   - 添加批量归一化

3. **超参数优化**
   - 调整学习率调度策略
   - 尝试不同的优化器（SGD, AdamW）
   - 调整Dropout比率

4. **性能提升**
   - 增加训练数据
   - 处理类别不平衡问题
   - 集成多个模型

## 常见问题

**Q: CUDA out of memory错误**
A: 减小BATCH_SIZE（如改为32或16）

**Q: 准确率较低**
A: 
- 检查数据是否正确加载
- 增加训练轮数
- 尝试数据增强
- 使用更复杂的模型架构

**Q: 如何查看训练进度**
A: 训练过程中会实时打印每个epoch的损失和准确率

## 参考资料

- FER2013数据集: https://www.kaggle.com/datasets/msambare/fer2013
- PyTorch文档: https://pytorch.org/docs/
- Streamlit文档: https://docs.streamlit.io/

## 作者

人脸情感识别项目
