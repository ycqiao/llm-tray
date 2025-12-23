"""
Rerank 服务 - 使用 FastAPI 提供本地 rerank API
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
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
    normalize: Optional[bool] = False  # 是否对分数进行归一化


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

    @staticmethod
    def normalize_scores(results: List[dict], method: str = "min_max") -> List[dict]:
        """
        对 rerank 结果进行分数归一化

        Args:
            results: rerank 结果列表，每个元素包含 index, document, score
            method: 归一化方法，"min_max" 或 "softmax"

        Returns:
            归一化后的结果列表，分数范围在 [0, 1]
        """
        if not results:
            return results

        scores = [r["score"] for r in results]
        scores_array = np.array(scores)

        if method == "min_max":
            # Min-Max 归一化：将分数映射到 [0, 1]
            min_score = scores_array.min()
            max_score = scores_array.max()
            if max_score == min_score:
                # 所有分数相同，归一化为 0.5
                normalized_scores = np.full_like(scores_array, 0.5)
            else:
                normalized_scores = (scores_array - min_score) / (max_score - min_score)
        elif method == "softmax":
            # Softmax 归一化：转换为概率分布
            # 使用温度参数来调整分布的尖锐程度
            temperature = 1.0
            exp_scores = np.exp(scores_array / temperature)
            normalized_scores = exp_scores / exp_scores.sum()
        else:
            raise ValueError(f"不支持的归一化方法: {method}")

        # 更新结果中的分数
        normalized_results = []
        for i, result in enumerate(results):
            normalized_result = result.copy()
            normalized_result["score"] = float(normalized_scores[i])
            normalized_results.append(normalized_result)

        return normalized_results


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
        "documents": ["文档1", "文档2", "文档3"],
        "normalize": true  # 可选，是否对分数进行归一化到 [0, 1]
    }

    响应格式：
    {
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 1, "relevance_score": 0.87}
        ]
    }

    注意：
    - normalize 参数为 true 时，分数会被归一化到 [0, 1] 范围
    - normalize 参数为 false 或未指定时，返回原始模型分数
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

        # 如果启用归一化，对分数进行归一化
        results = RerankModel.normalize_scores(results, method="softmax")

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

