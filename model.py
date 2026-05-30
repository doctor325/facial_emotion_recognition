import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets, models
import os

# 迁移学习模型 - 使用预训练的ResNet18
class EmotionResNet18(nn.Module):
    def __init__(self, num_classes=7, use_pretrained=True):
        super(EmotionResNet18, self).__init__()
        
        # 加载预训练的ResNet18 - 使用DEFAULT权重避免哈希校验问题
        if use_pretrained:
            try:
                # 尝试使用DEFAULT权重(更稳定)
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                print("✓ 成功加载ResNet18预训练权重(DEFAULT)")
            except Exception as e:
                print(f"加载DEFAULT权重失败: {e}")
                try:
                    # 备选方案: 使用IMAGENET1K_V2
                    self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V2)
                    print("✓ 成功加载ResNet18预训练权重(IMAGENET1K_V2)")
                except Exception as e2:
                    print(f"加载IMAGENET1K_V2权重失败: {e2}")
                    print("⚠ 使用无预训练权重的ResNet18")
                    self.backbone = models.resnet18(weights=None)
        else:
            self.backbone = models.resnet18(weights=None)
            print("使用无预训练权重的ResNet18")
        
        # 修改第一层卷积以接受单通道灰度图像
        # ResNet18原始输入是3通道(RGB),我们改为1通道(灰度)
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1,  # 改为单通道
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # 如果有预训练权重,复制第一通道的权重到新的卷积层
        if use_pretrained and original_conv.weight.data.shape[1] == 3:
            # 取RGB三个通道的平均值作为灰度通道权重
            self.backbone.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            print("✓ 已将预训练权重从3通道转换为1通道")
        
        # 替换最后的全连接层 - 增加正则化
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.6),  # 增加Dropout (0.5→0.6)
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # 增加Dropout (0.3→0.5)
            nn.Linear(256, num_classes)
        )
        
        print(f"模型输入通道数: 1 (灰度图)")
        print(f"输出类别数: {num_classes}")
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_trainable_layers(self, freeze_backbone=False):
        """获取可训练层,支持冻结主干网络"""
        if freeze_backbone:
            # 冻结主干网络,只训练分类头
            for param in self.backbone.parameters():
                param.requires_grad = False
            # 只训练最后的全连接层
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
            print("策略: 冻结主干网络,仅训练分类头")
        else:
            # 所有层都可训练(微调)
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("策略: 全模型微调")
        
        return self

# 保留原始CNN模型(可选使用)
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # 第一层卷积 (1 -> 32)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 添加批归一化
        
        # 第二层卷积 (32 -> 64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 第三层卷积 (64 -> 128)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # 计算全连接层输入大小
        # 48x48 -> 24x24 -> 12x12 -> 6x6
        # 128 * 6 * 6 = 4608
        self.fc1 = nn.Linear(128 * 6 * 6, 256)  # 减小全连接层
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv1 + BN + ReLU + Pool
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Conv2 + BN + ReLU + Pool
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Conv3 + BN + ReLU + Pool
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # 展平
        x = x.view(-1, 128 * 6 * 6)
        
        # Dropout
        x = self.dropout(x)
        
        # FC1 + BN + ReLU
        x = self.relu(self.bn_fc(self.fc1(x)))
        
        # Dropout
        x = self.dropout(x)
        
        # FC2 (输出层)
        x = self.fc2(x)
        
        return x

# 自定义高斯噪声变换(替代lambda,支持pickle序列化)
class GaussianNoise:
    """添加高斯噪声的数据增强变换"""
    def __init__(self, std=0.01):
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std
    
    def __repr__(self):
        return f'{self.__class__.__name__}(std={self.std})'

# 数据预处理和加载 - 增强版数据增强策略(解决过拟合)
def get_data_loaders(data_dir, batch_size=64, use_transfer_learning=True):
    # 训练数据增强 - 更强的数据增强(提高泛化能力)
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 确保转换为单通道灰度图
        transforms.Resize((54, 54)),  # 适度放大(52→54,增加多样性)
        transforms.RandomCrop(48),  # 随机裁剪回48x48
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(degrees=15),  # 提高旋转角度(12°→15°)
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1),  # 提高平移(0.08→0.1)
            scale=(0.85, 1.15)  # 提高缩放范围(0.9-1.1→0.85-1.15)
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 提高亮度对比度变化(0.15→0.2)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 验证和测试数据不需要增强
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)
    
    # 迁移学习通常使用较小的batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader

# Focal Loss - 处理类别不平衡
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 标签平滑损失
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        nll_loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# 训练函数 - 强正则化版(解决过拟合问题)
def train_model(model, train_loader, val_loader, num_epochs=60, learning_rate=0.001, 
                use_class_weights=False, use_focal_loss=False, focal_gamma=2.0,
                label_smoothing=0.1, freeze_backbone_epochs=5, weight_decay=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 第一阶段: 冻结主干网络,只训练分类头(延长冻结时间)
    if freeze_backbone_epochs > 0:
        print(f"\n{'='*60}")
        print(f"第一阶段: 冻结主干网络,训练分类头 ({freeze_backbone_epochs} epochs)")
        print(f"{'='*60}")
        model.get_trainable_layers(freeze_backbone=True)
        
        # 只优化分类头的参数
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=learning_rate * 2,  # 分类头使用较大学习率
            weight_decay=1e-3
        )
        
        # 使用标签平滑(降低平滑比例)
        if label_smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 2,
            epochs=freeze_backbone_epochs,
            steps_per_epoch=len(train_loader)
        )
        
        # 训练分类头
        for epoch in range(freeze_backbone_epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct_train / total_train
            print(f'Phase1 Epoch [{epoch+1}/{freeze_backbone_epochs}] '
                  f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
    
    # 第二阶段: 全模型微调
    print(f"\n{'='*60}")
    print(f"第二阶段: 全模型微调")
    print(f"{'='*60}")
    model.get_trainable_layers(freeze_backbone=False)
    
    # 使用不同的学习率(主干网络用更小学习率,分类头用大学习率)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'fc' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': learning_rate * 0.05},  # 主干网络更小学习率
        {'params': head_params, 'lr': learning_rate}  # 分类头大学习率
    ], weight_decay=weight_decay)  # 使用传入的weight_decay参数
    
    # 选择损失函数(增强Focal Loss)
    if use_focal_loss:
        # 计算类别权重用于Focal Loss
        class_sample_counts = [0] * 7
        for _, label in train_loader.dataset:
            class_sample_counts[label] += 1
        class_weights = 1.0 / torch.tensor(class_sample_counts, dtype=torch.float32)
        class_weights = (class_weights / class_weights.sum() * len(class_weights)).to(device)
        criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights)  # gamma: 2.0→3.0
        print(f"使用Focal Loss (gamma={focal_gamma})")
    elif use_class_weights:
        class_sample_counts = [0] * 7
        for _, label in train_loader.dataset:
            class_sample_counts[label] += 1
        class_weights = 1.0 / torch.tensor(class_sample_counts, dtype=torch.float32)
        class_weights = (class_weights / class_weights.sum() * len(class_weights)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"使用加权CrossEntropyLoss")
    else:
        if label_smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            print(f"使用Label Smoothing (smoothing={label_smoothing})")
        else:
            criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器 - 基于验证准确率动态调整(避免衰减过快)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # 监控验证准确率
        factor=0.5,  # 衰减因子
        patience=8,  # 耐心值
        min_lr=1e-6,  # 最小学习率
        verbose=False
    )
    
    # 存储训练历史
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    patience = 30  # 增加耐心值(25→30)
    patience_counter = 0
    
    # 计算第二阶段的训练轮数
    total_epochs = num_epochs - freeze_backbone_epochs
    
    for epoch in range(total_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # 更新学习率(传入验证准确率)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch [{epoch+1}/{total_epochs}] '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% '
              f'LR: {current_lr:.6f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
            print(f'  -> 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)')
        else:
            patience_counter += 1
            
        # 早停机制
        if patience_counter >= patience:
            print(f'\n早停触发 at epoch {epoch+1} (patience={patience})')
            print(f'最佳验证准确率: {best_val_acc:.2f}%')
            break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }