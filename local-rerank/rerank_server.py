"""
Rerank 服务 - 使用 FastAPI 提供本地 rerank API
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

app = FastAPI(title="Rerank Service", version="1.0.0")


class RerankRequest(BaseModel):
    """Rerank 请求模型"""
    query: str
    documents: List[str]
    top_k: Optional[int] = None  # 返回 top_k 个结果，None 表示返回全部


class RerankResponse(BaseModel):
    """Rerank 响应模型"""
    results: List[dict]  # [{"index": int, "document": str, "score": float}]


class OllamaRerankRequest(BaseModel):
    """Ollama 格式的 Rerank 请求模型"""
    model: str
    query: str
    documents: List[str]


class OllamaRerankResult(BaseModel):
    """Ollama 格式的 Rerank 结果"""
    index: int
    relevance_score: float


class OllamaRerankResponse(BaseModel):
    """Ollama 格式的 Rerank 响应模型"""
    results: List[OllamaRerankResult]


class RerankModel:
    """Rerank 模型封装类"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        初始化 rerank 模型

        Args:
            model_name: HuggingFace 模型名称
        """
        print(f"正在加载模型: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # 设置为评估模式
        print("模型加载完成！")

    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[dict]:
        """
        对文档进行 rerank

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前 k 个结果，None 表示返回全部

        Returns:
            排序后的结果列表，每个元素包含 index, document, score
        """
        if not documents:
            return []

        results = []

        # 批量处理以提高效率
        with torch.no_grad():
            for idx, doc in enumerate(documents):
                inputs = self.tokenizer(
                    query,
                    doc,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                score = self.model(**inputs).logits[0].item()
                results.append({
                    "index": idx,
                    "document": doc,
                    "score": score
                })

        # 按分数降序排序
        results.sort(key=lambda x: x["score"], reverse=True)

        # 如果指定了 top_k，只返回前 k 个
        if top_k is not None and top_k > 0:
            results = results[:top_k]

        return results


# 全局模型实例
rerank_model = None


@app.on_event("startup")
async def load_model():
    """启动时加载模型"""
    global rerank_model
    rerank_model = RerankModel()


@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "service": "Rerank Service",
        "version": "1.0.0",
        "status": "running",
        "model": "BAAI/bge-reranker-v2-m3"
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank API 端点

    接收查询和文档列表，返回按相关性分数排序的结果
    """
    if rerank_model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    if not request.documents:
        raise HTTPException(status_code=400, detail="文档列表不能为空")

    try:
        results = rerank_model.rerank(
            query=request.query,
            documents=request.documents,
            top_k=request.top_k
        )

        return RerankResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@app.post("/api/rerank", response_model=OllamaRerankResponse)
async def ollama_rerank(request: OllamaRerankRequest):
    """
    Ollama 兼容格式的 Rerank API 端点

    兼容 Ollama API 格式：
    POST /api/rerank
    {
        "model": "qllama/bge-reranker-v2-m3:latest",
        "query": "查询文本",
        "documents": ["文档1", "文档2", "文档3"]
    }

    响应格式：
    {
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 1, "relevance_score": 0.87}
        ]
    }
    """
    if rerank_model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    if not request.documents:
        raise HTTPException(status_code=400, detail="文档列表不能为空")

    try:
        # 使用内部的 rerank 方法
        results = rerank_model.rerank(
            query=request.query,
            documents=request.documents,
            top_k=None  # Ollama 格式返回所有结果
        )

        # 转换为 Ollama 格式
        ollama_results = [
            OllamaRerankResult(
                index=result["index"],
                relevance_score=result["score"]
            )
            for result in results
        ]

        return OllamaRerankResponse(results=ollama_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@app.post("/rerank/batch")
async def rerank_batch(requests: List[RerankRequest]):
    """
    批量 Rerank API 端点

    接收多个 rerank 请求，批量处理
    """
    if rerank_model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    results = []
    for request in requests:
        if not request.documents:
            results.append({"error": "文档列表不能为空"})
            continue

        try:
            rerank_results = rerank_model.rerank(
                query=request.query,
                documents=request.documents,
                top_k=request.top_k
            )
            results.append({"results": rerank_results})
        except Exception as e:
            results.append({"error": str(e)})

    return {"batch_results": results}


if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

