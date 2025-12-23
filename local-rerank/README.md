# Rerank 服务

基于 HuggingFace Transformers 的本地 Rerank 服务，使用 FastAPI 提供 HTTP API。

## 功能特性

- 🚀 基于 FastAPI 的高性能 HTTP 服务
- 🔥 支持批量 rerank 处理
- 📊 返回相关性分数和排序结果
- 🎯 支持 top_k 参数限制返回结果数量
- 🔌 兼容 Ollama API 格式

## 安装依赖

本项目使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理，并指定使用 Python 3.12.8。

### 安装 uv

```bash
pip install uv
```

### 使用 uv 安装依赖

uv 会自动创建虚拟环境并使用 Python 3.12.8：

```bash
cd local-rerank
uv sync
```

或者使用 `uv pip install`：

```bash
uv pip install -r requirements.txt
```

### 使用 uv 运行

安装依赖后，可以使用 uv 运行服务：

```bash
uv run python rerank_server.py
```

或使用 uvicorn：

```bash
uv run uvicorn rerank_server:app --host 0.0.0.0 --port 8000
```

## 启动服务

### 方式一：使用 uv 运行（推荐）

```bash
cd local-rerank
uv run python rerank_server.py
```

### 方式二：使用 uv 运行 uvicorn

```bash
cd local-rerank
uv run uvicorn rerank_server:app --host 0.0.0.0 --port 8000
```

### 方式三：激活虚拟环境后运行

```bash
cd local-rerank
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows

python rerank_server.py
# 或
uvicorn rerank_server:app --host 0.0.0.0 --port 8000
```

服务启动后，默认运行在 `http://localhost:8000`

## API 文档

启动服务后，访问以下地址查看自动生成的 API 文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 端点

### 1. 健康检查

```bash
GET /health
```

### 2. Rerank 单个请求

```bash
POST /rerank
Content-Type: application/json

{
  "query": "如何实现 RAG",
  "documents": [
    "RAG 是一种结合检索与生成的技术",
    "今天天气很好",
    "RAG 通过检索相关文档来增强生成模型的能力"
  ],
  "top_k": 3  # 可选，返回前3个结果
}
```

响应：

```json
{
  "results": [
    {
      "index": 0,
      "document": "RAG 是一种结合检索与生成的技术",
      "score": 8.234
    },
    {
      "index": 2,
      "document": "RAG 通过检索相关文档来增强生成模型的能力",
      "score": 7.891
    }
  ]
}
```

### 3. Ollama 兼容格式的 Rerank

```bash
POST /api/rerank
Content-Type: application/json

{
  "model": "qllama/bge-reranker-v2-m3:latest",
  "query": "查询文本",
  "documents": ["文档1", "文档2", "文档3"]
}
```

响应：

```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 8.234
    },
    {
      "index": 1,
      "relevance_score": 7.891
    },
    {
      "index": 2,
      "relevance_score": 2.345
    }
  ]
}
```

**注意**：`model` 参数在当前实现中会被忽略，实际使用的是服务启动时加载的模型。此端点完全兼容 Ollama API 格式。

### 4. 批量 Rerank

```bash
POST /rerank/batch
Content-Type: application/json

[
  {
    "query": "如何实现 RAG",
    "documents": ["文档1", "文档2"],
    "top_k": 2
  },
  {
    "query": "Python 编程",
    "documents": ["文档3", "文档4"],
    "top_k": 1
  }
]
```

## 使用示例

### cURL 示例

**标准格式：**
```bash
curl -X POST "http://localhost:8000/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何实现 RAG",
    "documents": [
      "RAG 是一种结合检索与生成的技术",
      "今天天气很好"
    ],
    "top_k": 2
  }'
```

**Ollama 兼容格式：**
```bash
curl http://localhost:8000/api/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qllama/bge-reranker-v2-m3:latest",
    "query": "查询文本",
    "documents": ["文档1", "文档2", "文档3"]
  }'
```

### Python 测试示例

使用 `test_ollama_rerank.py` 测试 Ollama 兼容格式的 API：

```bash
uv run python test_ollama_rerank.py
```

或激活虚拟环境后：

```bash
source .venv/bin/activate
python test_ollama_rerank.py
```

## 配置

### 修改模型

在 `rerank_server.py` 中修改模型名称：

```python
rerank_model = RerankModel(model_name="BAAI/bge-reranker-v2-m3")
```

### 修改端口

在启动时指定端口：

```bash
uvicorn rerank_server:app --host 0.0.0.0 --port 8080
```

或在代码中修改：

```python
uvicorn.run(app, host="0.0.0.0", port=8080)
```

## 性能优化建议

1. **批量处理**：使用 `/rerank/batch` 端点进行批量处理
2. **GPU 加速**：如果有 GPU，模型会自动使用 GPU
3. **并发处理**：使用多个 worker 进程：

```bash
uvicorn rerank_server:app --host 0.0.0.0 --port 8000 --workers 4
```

## 注意事项

- 首次运行时会自动下载模型，需要一定时间
- 模型会占用一定的内存（约 1-2GB）
- 建议在有 GPU 的环境下运行以获得更好的性能

