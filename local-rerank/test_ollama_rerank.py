"""
测试 Ollama 兼容格式的 Rerank API
"""
import requests
import json

def test_ollama_rerank():
    """测试 Ollama 格式的 rerank API"""
    url = "http://localhost:8000/api/rerank"
    payload = {
        "model": "qllama/bge-reranker-v2-m3:latest",
        "query": "查询文本",
        "documents": ["文档1", "文档2", "文档3"]
    }

    print("请求 URL:", url)
    print("请求体:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\n发送请求...")

    response = requests.post(url, json=payload)

    print(f"\n状态码: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("响应:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        print("\n解析结果:")
        for i, res in enumerate(result.get("results", []), 1):
            print(f"{i}. 索引: {res['index']}, 相关性分数: {res['relevance_score']:.4f}")
    else:
        print("错误响应:")
        print(response.text)


def test_ollama_rerank_curl_format():
    """使用 curl 格式的测试（实际调用）"""
    import subprocess

    curl_command = [
        "curl",
        "http://localhost:8000/api/rerank",
        "-d",
        json.dumps({
            "model": "qllama/bge-reranker-v2-m3:latest",
            "query": "查询文本",
            "documents": ["文档1", "文档2", "文档3"]
        }),
        "-H",
        "Content-Type: application/json"
    ]

    print("执行 curl 命令:")
    print(" ".join(curl_command))
    print("\n结果:")

    try:
        result = subprocess.run(curl_command, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("错误:", result.stderr)
    except Exception as e:
        print(f"执行失败: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("测试 Ollama 兼容格式的 Rerank API")
    print("=" * 60)

    # 使用 requests 库测试
    test_ollama_rerank()

    print("\n" + "=" * 60)
    print("使用 curl 命令测试（可选）")
    print("=" * 60)
    print("\n可以直接使用以下 curl 命令:")
    print("curl http://localhost:8000/api/rerank -d '{\"model\": \"qllama/bge-reranker-v2-m3:latest\", \"query\": \"查询文本\", \"documents\": [\"文档1\", \"文档2\", \"文档3\"]}' -H 'Content-Type: application/json'")

