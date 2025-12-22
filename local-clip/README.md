# CLIP 图像分类器

使用 OpenAI CLIP 模型进行零样本图像分类的 Python 项目。

## 功能特性

- 使用 CLIP (Contrastive Language-Image Pre-training) 模型进行零样本图像分类
- 支持从 URL 加载图像
- 可视化分类结果
- 支持 GPU 加速（如果可用）

## 项目结构

```
.
├── main.py              # 主程序文件
├── pyproject.toml       # 项目配置文件
└── README.md           # 项目说明文档
```

## 安装

本项目使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理。

### 1. 安装 uv（如果尚未安装）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装 Python 3.12.8:
```bash
uv python install 3.12.8
```

### 2. 安装项目依赖

```bash
uv sync --python 3.12.8
```

### 3. GPU 支持（可选）

如果需要 GPU 加速，可以安装特定版本的 PyTorch：

**CUDA 11.7:**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

**CUDA 11.8:**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 使用方法

### 运行主程序

```bash
uv run main.py
```

或者激活虚拟环境后运行：

```bash
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows

python main.py
```

## 依赖项

- `torch`: PyTorch 深度学习框架
- `torchvision`: 计算机视觉工具库
- `transformers`: Hugging Face 的 transformers 库（包含 CLIP 模型）
- `pillow`: 图像处理库
- `matplotlib`: 数据可视化库
- `requests`: HTTP 请求库
- `certifi`: SSL 证书支持

## 示例输出

程序会：
1. 从指定 URL 下载图像
2. 使用 CLIP 模型对图像进行分类
3. 输出每个类别的概率
4. 生成可视化结果并保存为 `clip_result.png`
5. 显示预测的类别

## 注意事项

- 首次运行时会自动下载 CLIP 模型（约 500MB），需要网络连接
- 模型会缓存在本地，后续运行无需重新下载
- 如果系统支持 GPU，模型会自动使用 GPU 加速
- 图片会保存到 `clip_result.png`，可在文件管理器中打开查看

## 许可证

本项目使用 MIT 许可证。
