# Rerank & Embedding API Service

🇯🇵 日本語 | [🇺🇸 English](README.md)

OpenAI互換のRerank/Embedding APIサービス。BGE RerankerモデルとEmbeddingモデルを使用した文書の再ランキングとテキスト埋め込み生成機能を提供します。

## 機能

- **OpenAI API互換**: RerankおよびEmbeddingエンドポイント
- **日本語対応**: 日本語専用の高性能モデルをサポート
- **多言語対応**: 100以上の言語に対応したモデル
- **動的モデル選択**: APIリクエストによるモデル選択
- **マルチGPUサポート**: NVIDIA CUDA、AMD ROCm、CPU自動検出
- **Dockerデプロイメント**: 複数のDocker設定ファイル
- **高速レスポンス**: 非同期処理とモデルキャッシュ
- **プロキシ対応**: 企業環境向けHTTP/HTTPSプロキシサポート

## サポートモデル

### Reranking（再ランキング）モデル

#### 日本語専用モデル

| モデル名 | 短縮名 | 最大長 | サイズ | 説明 |
|---------|-------|-------|------|------|
| hotchpotch/japanese-reranker-cross-encoder-large-v1 | japanese-reranker-large | 512 | 334MB | 日本語最高性能 |
| hotchpotch/japanese-reranker-cross-encoder-base-v1 | japanese-reranker-base | 512 | 111MB | 日本語バランス型 |
| hotchpotch/japanese-reranker-cross-encoder-small-v1 | japanese-reranker-small | 512 | 67MB | 高速推論 |
| hotchpotch/japanese-bge-reranker-v2-m3-v1 | japanese-bge-v2-m3 | 8192 | ~500MB | 日本語特化版 |

#### 多言語モデル

| モデル名 | 短縮名 | 最大長 | サイズ | 説明 |
|---------|-------|-------|------|------|
| jinaai/jina-reranker-v2-base-multilingual | jina-reranker-v2-multilingual | 1024 | 278MB | 100+言語対応 **デフォルト** |
| BAAI/bge-reranker-v2-m3 | bge-reranker-v2-m3 | 32000 | ~600MB | 32kトークン対応 |
| Alibaba-NLP/gte-multilingual-reranker-base | gte-multilingual-reranker | 8192 | 560MB | 70+言語対応 |
| mixedbread-ai/mxbai-rerank-large-v1 | mxbai-rerank-large | 8192 | 1.5GB | 高性能 |
| Cohere/rerank-multilingual-v3.0 | cohere-rerank-multilingual | 4096 | ~400MB | 商用グレード |

### Embedding（埋め込み）モデル

#### 日本語専用モデル

| モデル名 | 短縮名 | 最大長 | 次元 | 説明 |
|---------|-------|-------|-----|------|
| cl-nagoya/ruri-large | ruri-large | 512 | 1024 | JMTEB最高性能 |
| cl-nagoya/ruri-base | ruri-base | 512 | 768 | 日本語バランス型 |
| MU-Kindai/Japanese-SimCSE-BERT-large-unsup | japanese-simcse-large | 512 | 1024 | 教師なし学習 |
| sonoisa/sentence-luke-japanese-base-lite | luke-japanese-base | 512 | 768 | 知識強化型 |
| pkshatech/GLuCoSE-base-ja-v2 | glucose-ja-v2 | 512 | 768 | 企業開発 |

#### 多言語モデル

| モデル名 | 短縮名 | 最大長 | 次元 | 説明 |
|---------|-------|-------|-----|------|
| BAAI/bge-m3 | bge-m3 | 8192 | 1024 | **デフォルト** |
| nvidia/NV-Embed-v2 | nv-embed-v2 | 32768 | 4096 | SOTA性能 |
| intfloat/e5-mistral-7b-instruct | e5-mistral-7b | 32768 | 4096 | 高品質 |
| mixedbread-ai/mxbai-embed-large-v1 | mxbai-embed-large | 512 | 1024 | 本番運用 |

## クイックスタート

### Docker使用

```bash
# 基本構築と実行
docker build -t rerank-api .
docker run -d --name rerank-api -p 7987:7987 --gpus all rerank-api

# プロキシ設定付きビルド
docker build \
  --build-arg HTTP_PROXY=http://proxy.company.com:8080 \
  --build-arg HTTPS_PROXY=http://proxy.company.com:8080 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  -t rerank-api .

# AMD GPU用
docker build -f docker/Dockerfile.amd -t rerank-api:amd .

# CPU専用
docker build -f docker/Dockerfile.flexible --build-arg COMPUTE_MODE=cpu -t rerank-api:cpu .
```

### Docker Compose

```bash
# NVIDIA GPU
docker-compose up -d

# プロキシ設定
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
export NO_PROXY=localhost,127.0.0.1
docker-compose up -d

# AMD GPU
docker-compose -f docker/docker-compose.amd.yml up -d

# CPU専用
docker-compose -f docker/docker-compose.cpu.yml up -d
```

### ローカル開発

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

## API使用方法

### 利用可能なモデル確認

```bash
curl http://localhost:7987/models
```

### Rerank API

#### 日本語高性能モデル使用

```bash
curl -X POST "http://localhost:7987/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "japanese-reranker-large",
    "query": "人工知能とは何ですか？",
    "documents": [
      "AIは機械で人間の知能を模倣します。",
      "明日の天気予報は雨です。",
      "機械学習はAI技術の一部です。"
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

#### 日本語バランス型モデル使用

```bash
curl -X POST "http://localhost:7987/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "japanese-reranker-base",
    "query": "自然言語処理",
    "documents": [
      "NLPはコンピュータが人間の言語を理解するのを助けます。",
      "パスタを茹でるにはまずお湯を沸かします。",
      "テキスト解析はNLPの中核的な要素です。"
    ]
  }'
```

#### 多言語モデル使用

```bash
curl -X POST "http://localhost:7987/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-reranker-v2-multilingual",
    "query": "機械学習アルゴリズム",
    "documents": [
      "機械学習は人工知能の一分野です。",
      "今日は良い天気です。",
      "ディープラーニングは機械学習の手法の一つです。"
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

### Embedding API

#### 日本語高性能モデル使用

```bash
curl -X POST "http://localhost:7987/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ruri-large",
    "input": "自然言語処理は人工知能の重要な分野です。"
  }'
```

#### 日本語バランス型モデル使用

```bash
curl -X POST "http://localhost:7987/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ruri-base",
    "input": "日本語のテキスト埋め込みを生成します。"
  }'
```

#### 多言語デフォルトモデル使用

```bash
curl -X POST "http://localhost:7987/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": [
      "最初の埋め込み対象テキスト",
      "2番目の埋め込み対象テキスト",
      "3番目の埋め込み対象テキスト"
    ]
  }'
```

## Docker設定

### Dockerファイル構造

```
docker/
├── Dockerfile                  # 標準NVIDIA GPU用
├── Dockerfile.amd             # AMD ROCm GPU用
├── Dockerfile.flexible        # CPU/GPU柔軟対応
├── docker-compose.yml         # 標準compose
├── docker-compose.amd.yml     # AMD GPU compose
├── docker-compose.cpu.yml     # CPU専用compose
├── requirements.txt           # 標準要件
├── requirements.amd.txt       # AMD専用要件
└── requirements-cpu.txt       # CPU専用要件
```

**注意**: メインの`docker-compose.yml`は便利のためルートディレクトリにも配置されています。

### ビルド引数

```bash
# プロキシサポート付きビルド
docker build \
  --build-arg HTTP_PROXY=http://proxy.company.com:8080 \
  --build-arg HTTPS_PROXY=http://proxy.company.com:8080 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  -t rerank-api .

# AMD GPU版ビルド
docker build -f docker/Dockerfile.amd -t rerank-api:amd .

# CPU専用版ビルド
docker build -f docker/Dockerfile.flexible \
  --build-arg COMPUTE_MODE=cpu \
  -t rerank-api:cpu .
```

## 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| HOST | 0.0.0.0 | サービスホスト |
| PORT | 7987 | サービスポート |
| WORKERS | 1 | ワーカー数 |
| RERANKER_MODEL_NAME | jinaai/jina-reranker-v2-base-multilingual | デフォルトrerankerモデル |
| RERANKER_MODELS_DIR | /app/models | モデル保存ディレクトリ |
| EMBEDDING_MODEL_NAME | BAAI/bge-m3 | デフォルトembeddingモデル |
| HTTP_PROXY | - | HTTPプロキシサーバーURL |
| HTTPS_PROXY | - | HTTPSプロキシサーバーURL |
| NO_PROXY | - | プロキシをバイパスするホストのリスト |

## テスト

### テスト実行

```bash
# 全テスト実行
pytest tests/

# カバレッジ付き
pytest tests/ --cov=.

# 個別テスト
python -m pytest tests/test_api.py -v

# API例テスト
python tests/test_api_example.py

# ハードウェア検出テスト
bash tests/test_detection.sh
```

### Docker設定テスト

```bash
# 異なるDocker設定のテスト
docker-compose up -d                                      # NVIDIA GPU（ルート）
docker-compose -f docker/docker-compose.yml up -d         # NVIDIA GPU（docker/）
docker-compose -f docker/docker-compose.amd.yml up -d     # AMD GPU
docker-compose -f docker/docker-compose.cpu.yml up -d     # CPU専用

# 特定Dockerファイルでのテスト
docker build -f docker/Dockerfile.amd -t test:amd .
docker build -f docker/Dockerfile.flexible --build-arg COMPUTE_MODE=cpu -t test:cpu .
```

### コード品質

```bash
# フォーマット
black .

# Lint
ruff check .

# 型チェック
mypy app.py
```

### 手動APIテスト

付属のテストスクリプトを使用：

```bash
python tests/test_api_example.py
```

## パフォーマンス最適化

### GPU設定

#### NVIDIA GPU サポート
- NVIDIAドライバー（最新版推奨）
- CUDA 11.8以上対応
- GPU memory 4GB以上推奨
- Docker用NVIDIA Container Toolkit

#### AMD GPU サポート  
- ROCm 6.0以上対応
- AMD GPUドライバー（AMDGPU-PROまたはオープンソース）
- GPU memory 4GB以上推奨
- AMD GPU デバイスアクセス用Docker設定（`/dev/kfd`、`/dev/dri`）

#### 自動検出機能
サービスは利用可能なGPUハードウェアを自動検出：
- 🟢 NVIDIA GPU → CUDA加速を使用
- 🔵 AMD GPU → ROCm加速を使用  
- ⚪ GPU無し → CPUにフォールバック

### メモリ管理

- 効率的なモデルキャッシュ
- スループット向上のためのバッチ処理
- 設定可能なワーカー数
- 自動メモリクリーンアップ

## トラブルシューティング

### GPUが認識されない場合

CPUモードを使用：
```bash
# docker-compose使用
docker-compose -f docker/docker-compose.cpu.yml up -d

# docker run使用
docker run -d --name rerank-api \
  -p 7987:7987 \
  -e CUDA_VISIBLE_DEVICES=-1 \
  rerank-api
```

GPUサポートの確認：
```bash
# GPU テスト
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### プロキシ環境での使用

プロキシ設定の環境変数を設定：
```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
export NO_PROXY=localhost,127.0.0.1
docker-compose up -d
```

## API使用例

### Pythonクライアント

```python
import requests

def rerank_documents(query, documents, model="japanese-reranker-large"):
    response = requests.post("http://localhost:7987/v1/rerank", json={
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": 5,
        "return_documents": True
    })
    return response.json()

def get_embeddings(texts, model="ruri-large"):
    response = requests.post("http://localhost:7987/v1/embeddings", json={
        "model": model,
        "input": texts
    })
    return response.json()

# 使用例
query = "機械学習アルゴリズム"
docs = [
    "機械学習は人工知能の一分野です",
    "今日の天気は晴れで暖かいです",
    "ニューラルネットワークは強力なMLアルゴリズムです",
    "料理には新鮮な食材が必要です"
]

# Reranking
results = rerank_documents(query, docs)
for result in results["results"]:
    print(f"スコア: {result['relevance_score']:.3f} - {result['document']}")

# Embeddings
embeddings = get_embeddings(["自然言語処理", "機械学習"])
print(f"埋め込み次元: {len(embeddings['data'][0]['embedding'])}")
```

## カスタムモデル

カスタムモデルを追加するには、`model_loader.py`と`embedding_loader.py`の`supported_models`辞書を更新してください：

```python
# Rerankerモデル追加例
self.supported_models = {
    "your-custom/reranker-model": {
        "name": "custom-reranker",
        "description": "カスタムリランカーモデル",
        "max_length": 512,
        "language": "japanese",
        "trust_remote_code": False
    }
}

# Embeddingモデル追加例
self.supported_models = {
    "your-custom/embedding-model": {
        "name": "custom-embedding",
        "description": "カスタム埋め込みモデル",
        "max_length": 512,
        "dimensions": 768,
        "language": "japanese"
    }
}
```

## ライセンス

このプロジェクトは MIT ライセンスのもとで公開されています。

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まず Issue を作成して変更内容を議論してください。

## サポート

問題が発生した場合は、GitHub の Issue ページで報告してください。

---

**注意**: このサービスは日本語を含む多言語での文書再ランキングと埋め込み生成機能を提供し、適切な監視とスケーリングを考慮した本番環境での使用を想定して設計されています。