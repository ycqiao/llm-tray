import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，确保在任何环境下都能保存图片
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_clip_model():
    """
    加载CLIP模型和处理器
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def get_image_from_url(url):
    """
    从URL获取图像
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        return Image.open(response.raw)
    else:
        raise Exception(f"无法获取图像，状态码：{response.status_code}")

def get_image_from_fs(path):
    """
    从文件系统获取图像
    """
    return Image.open(path)

def classify_image(model, processor, image, categories):
    """
    使用CLIP对图像进行零样本分类
    """
    # 将类别转换为提示文本
    texts = [f"一张{category}的照片" for category in categories]

    # 处理输入
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    # 计算图像-文本相似度
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取相似度分数
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return probs.numpy()[0]

def visualize_results(image, categories, probs):
    """
    可视化CLIP分类结果
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 显示图像
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title("输入图像", fontsize=14)

    # 显示分类概率
    y_pos = np.arange(len(categories))
    bars = ax2.barh(y_pos, probs, align='center', color='steelblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories, fontsize=12)
    ax2.set_xlabel('概率', fontsize=12)
    ax2.set_title('CLIP零样本分类结果', fontsize=14)
    ax2.set_xlim(0, max(probs) * 1.1)

    # 在条形图上添加数值标签
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{prob:.2%}', ha='left' if width < max(probs) * 0.1 else 'right',
                va='center', fontsize=10)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_path = "clip_result.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n结果图片已保存到: {os.path.abspath(output_path)}")

    plt.close(fig)  # 关闭figure以释放内存

# 主程序
def main():
    # 加载模型
    model, processor = load_clip_model()

    # 获取示例图像
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    # image = get_image_from_url(url)
    image = get_image_from_fs("dog.png")

    # 定义要识别的类别
    categories = ["狗", "猫", "鸟", "鱼", "马", "汽车", "自行车", "飞机"]

    # 进行分类
    probs = classify_image(model, processor, image, categories)

    # 输出结果
    print("\nCLIP零样本分类结果:")
    for i, (category, prob) in enumerate(zip(categories, probs)):
        print(f"{category}: {prob:.2%}")

    # 可视化结果
    visualize_results(image, categories, probs)

    print(f"\n预测类别: {categories[np.argmax(probs)]}")

if __name__ == "__main__":
    main()
