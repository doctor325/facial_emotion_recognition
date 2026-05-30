import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from model import EmotionResNet18, get_data_loaders, train_model
import os

# 修复matplotlib中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_confusion_matrix(cm, class_names):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='混淆矩阵',
           ylabel='真实标签',
           xlabel='预测标签')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在每个单元格中显示数值
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=200, bbox_inches='tight')
    print("混淆矩阵已保存为 confusion_matrix.png")

def main():
    # 配置参数 - 强正则化版(解决过拟合问题)
    DATA_DIR = 'data'
    BATCH_SIZE = 32  # 迁移学习使用较小的batch size
    NUM_EPOCHS = 60  # 增加训练轮数 (55→60)
    LEARNING_RATE = 0.001  # 学习率
    FREEZE_EPOCHS = 5  # 降低冻结轮数 (7→5,避免冻结阶段过拟合)
    USE_FOCAL_LOSS = True  # 使用Focal Loss处理类别不平衡
    FOCAL_GAMMA = 2.0  # 降低gamma值 (2.5→2.0,减少过度关注困难样本)
    LABEL_SMOOTHING = 0.1  # 增加标签平滑 (0.08→0.1,增强正则化)
    WEIGHT_DECAY = 5e-4  # 增加权重衰减 (1e-4→5e-4,强正则化)
    
    # ========== 训练模式选择 ==========
    RESUME_TRAINING = False  # 是否从已有模型继续训练 (False=从头训练, True=继续训练)
    CHECKPOINT_PATH = 'best_model.pth'  # 继续训练时加载的模型路径
    REDUCED_LR_FOR_RESUME = True  # 继续训练时是否降低学习率(推荐True)
    # ==================================
    
    # 检查数据目录是否存在
    if not os.path.exists(DATA_DIR):
        print(f"错误: 未找到数据目录 {DATA_DIR}")
        print("请确保数据目录包含 train, val, test 子目录")
        return
    
    print("正在加载数据...")
    print("数据增强策略: 旋转±15°, 平移±10%, 缩放0.85-1.15 (增强版)")
    print("正则化策略: Dropout 0.6+0.5, Weight Decay 5e-4, Label Smoothing 0.1")
    print("目标: 解决过拟合问题(当前训练-验证差距16%)")
    train_loader, val_loader, test_loader = get_data_loaders(DATA_DIR, BATCH_SIZE, use_transfer_learning=True)
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 初始化模型
    print("\n初始化ResNet18迁移学习模型...")
    
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        # 模式A: 从已有模型继续训练
        print(f"\n{'='*60}")
        print(f"训练模式: 继续训练 (Resume Training)")
        print(f"加载检查点: {CHECKPOINT_PATH}")
        print(f"{'='*60}")
        
        model = EmotionResNet18(num_classes=7, use_pretrained=False)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        model.load_state_dict(checkpoint)
        print("✓ 成功加载已有模型权重")
        
        # 继续训练时降低学习率(推荐)
        if REDUCED_LR_FOR_RESUME:
            original_lr = LEARNING_RATE
            LEARNING_RATE = LEARNING_RATE * 0.1  # 降低10倍
            print(f"✓ 学习率已调整: {original_lr} → {LEARNING_RATE}")
        
        # 继续训练时跳过冻结阶段
        if FREEZE_EPOCHS > 0:
            print(f"⚠ 继续训练模式下,跳过冻结阶段(已有模型已适配)")
            FREEZE_EPOCHS = 0
    else:
        # 模式B: 从头开始训练
        if RESUME_TRAINING:
            print(f"\n{'='*60}")
            print(f"警告: 未找到检查点 {CHECKPOINT_PATH}")
            print(f"将改为从头开始训练")
            print(f"{'='*60}")
        
        print(f"\n{'='*60}")
        print(f"训练模式: 从头训练 (Train from Scratch)")
        print(f"将加载ImageNet预训练权重")
        print(f"{'='*60}")
        
        model = EmotionResNet18(num_classes=7, use_pretrained=True)
    
    print("\n模型结构:")
    print(model)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 训练模型
    print("\n开始训练...")
    print(f"{'='*60}")
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        print(f"训练策略: 继续微调 {NUM_EPOCHS} 轮 (已跳过冻结阶段)")
    else:
        print(f"训练策略: {FREEZE_EPOCHS}轮冻结训练 + {NUM_EPOCHS}轮全模型微调")
    print(f"Focal Loss: {'启用' if USE_FOCAL_LOSS else '禁用'} (gamma={FOCAL_GAMMA})")
    print(f"Label Smoothing: {LABEL_SMOOTHING}")
    print(f"学习率策略: 主干×0.05, 分类头×1.0")
    print(f"数据增强: 保守策略(避免过度增强)")
    print(f"{'='*60}")
    
    history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        use_focal_loss=USE_FOCAL_LOSS,
        focal_gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
        freeze_backbone_epochs=FREEZE_EPOCHS,
        weight_decay=WEIGHT_DECAY  # 新增权重衰减参数
    )
    
    # 绘制训练曲线
    print("\n绘制训练曲线...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(history['train_losses'], label='训练损失', linewidth=2)
    ax1.plot(history['val_losses'], label='验证损失', linewidth=2)
    ax1.set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
    ax1.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax1.set_ylabel('损失值 (Loss)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(history['train_accuracies'], label='训练准确率', linewidth=2, color='green')
    ax2.plot(history['val_accuracies'], label='验证准确率', linewidth=2, color='orange')
    ax2.set_title('训练和验证准确率曲线', fontsize=14, fontweight='bold')
    ax2.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("训练曲线已保存为 training_history.png")
    
    # 在测试集上评估
    print("\n在测试集上评估...")
    model.load_state_dict(torch.load('best_model.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算准确率
    test_accuracy = 100 * sum([1 for p, l in zip(all_preds, all_labels) if p == l]) / len(all_labels)
    print(f"\n测试集准确率: {test_accuracy:.2f}%")
    
    # 打印分类报告
    emotion_labels = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=emotion_labels))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, emotion_labels)
    
    print("\n训练完成!")
    print("最佳模型已保存为 best_model.pth")

if __name__ == "__main__":
    main()
