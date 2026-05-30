import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

# 导入模型类
sys.path.append(os.path.dirname(__file__))
from model import EmotionResNet18

# 加载模型
@st.cache_resource
def load_model():
    model = EmotionResNet18(num_classes=7, use_pretrained=False)
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# 图像预处理 - 与训练时保持一致
def preprocess_image(image):
    # 转换为灰度图
    if image.mode != 'L':
        image = image.convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)

# 情感标签映射
emotion_labels = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 主应用界面
def main():
    st.title("人脸情感识别系统")
    st.write("上传一张人脸图像，系统将识别其情感状态")
    
    # 上传图像
    uploaded_file = st.file_uploader("选择一张人脸图像", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 显示上传的图像
        image = Image.open(uploaded_file)
        st.image(image, caption='上传的图像', use_column_width=True)
        
        # 预测按钮
        if st.button('识别情感'):
            # 预处理图像
            processed_image = preprocess_image(image)
            
            # 加载模型
            model = load_model()
            
            # 进行预测
            with torch.no_grad():
                output = model(processed_image)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # 显示结果
            st.write(f"**预测结果**: {emotion_labels[predicted_class]}")
            st.write(f"**置信度**: {confidence:.2%}")
            
            # 可视化概率分布
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(emotion_labels, probabilities[0].numpy())
            ax.set_title('Emotion Type Probabilities')
            ax.set_xlabel('Emotion Types')
            ax.set_ylabel('Probability')
            st.pyplot(fig)

if __name__ == "__main__":
    main()