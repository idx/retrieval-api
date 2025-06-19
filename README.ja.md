# Retrieval API Service

🇯🇵 日本語 | [🇺🇸 English](README.md)

OpenAI互換のRetrieval APIサービス。統一されたモデル管理システムと日本語言語サポートを備えた、最先端の文書再ランキングとテキスト埋め込み生成機能を提供します。

## 機能

- **デュアルAPIサポート**: OpenAI互換のRerankおよびEmbeddingエンドポイント
- **統一モデル管理**: RerankerとEmbeddingの両方でプリロード、キャッシング、動的切り替え
- **日本語言語サポート**: 日本語テキスト処理に特化したモデル
- **多言語モデル**: 100以上の言語対応の高性能モデル
- **動的モデル選択**: 自動フォールバック機能付きAPIリクエストによるモデル切り替え
- **マルチGPUサポート**: NVIDIA CUDA、AMD ROCm自動検出
- **CPUフォールバック**: GPU依存なしでのシームレス動作
- **Dockerデプロイメント**: 複数のDocker設定による簡単デプロイメント
- **本番運用対応**: 非同期処理、メモリ管理、監視機能

## サポートモデル

### Reranking（再ランキング）モデル

#### 日本語専用モデル

| モデル名 | 短縮名 | 最大長 | サイズ | 説明 |
|---------|-------|-------|------|------|
| hotchpotch/japanese-reranker-cross-encoder-large-v1 | japanese-reranker-large | 512 | 334MB | 日本語最高性能 |
| hotchpotch/japanese-reranker-cross-encoder-base-v1 | japanese-reranker-base | 512 | 111MB | 日本語バランス型 |
| pkshatech/GLuCoSE-base-ja | glucose-base-ja | 512 | ~400MB | GLuCoSE日本語ベースモデル |

#### 多言語モデル

| モデル名 | 短縮名 | 最大長 | サイズ | 説明 |
|---------|-------|-------|------|------|
| maidalun1020/bce-reranker-base_v1 | bce-reranker-base_v1 | 512 | ~400MB | BGE Rerankerベースモデル v1 **デフォルト** |
| jinaai/jina-reranker-v2-base-multilingual | jina-reranker-v2 | 1024 | 278MB | Jina Reranker v2多言語（100以上の言語） |
| mixedbread-ai/mxbai-rerank-large-v1 | mxbai-rerank-large | 512 | 1.5GB | MixedBread AI Rerank Large v1（高性能） |

### Embedding（埋め込み）モデル

#### 日本語専用モデル

| モデル名 | 短縮名 | 最大長 | 次元 | 説明 |
|---------|-------|-------|-----|------|
| cl-nagoya/ruri-large | ruri-large | 512 | 768 | RURI Large日本語埋め込み（JMTEB最高性能） |
| cl-nagoya/ruri-base | ruri-base | 512 | 768 | RURI Base日本語埋め込み（日本語バランス型） |
| MU-Kindai/Japanese-SimCSE-BERT-large-unsup | japanese-simcse-large | 512 | 1024 | Japanese SimCSE BERT Large |
| sonoisa/sentence-luke-japanese-base-lite | sentence-luke-base | 512 | 768 | LUKE Japanese Base Lite |
| pkshatech/GLuCoSE-base-ja-v2 | glucose-base-ja-v2 | 512 | 768 | GLuCoSE Japanese v2 |

#### 多言語モデル

| モデル名 | 短縮名 | 最大長 | 次元 | 説明 |
|---------|-------|-------|-----|------|
| BAAI/bge-m3 | bge-m3 | 8192 | 1024 | BGE M3多言語埋め込み **デフォルト** |
| intfloat/multilingual-e5-large | multilingual-e5-large | 512 | 1024 | 多言語E5 Large（100以上の言語） |
| mixedbread-ai/mxbai-embed-large-v1 | mxbai-embed-large | 512 | 1024 | MixedBread AI Large v1 |
| nvidia/NV-Embed-v2 | nv-embed-v2 | 32768 | 4096 | NVIDIA NV-Embed v2（SOTA性能） |

## クイックスタート

### Dockerデプロイメント

#### 自動GPU/CPU検出

自動検出用の提供スクリプトを使用：

```bash
# スクリプトを実行可能にする
chmod +x start.sh

# 自動GPU/CPU検出で開始
./start.sh
```

#### 手動Dockerコマンド

```bash
# NVIDIA GPU用にビルド
docker build -t retrieval-api .

# プロキシサポート付きビルド
docker build -t retrieval-api \
  --build-arg HTTP_PROXY=http://proxy.company.com:8080 \
  --build-arg HTTPS_PROXY=http://proxy.company.com:8080 \
  --build-arg NO_PROXY=localhost,127.0.0.1 .

# AMD GPU用にビルド
docker build -f docker/Dockerfile.amd -t retrieval-api:amd .

# フレキシブル設定でビルド
docker build -f docker/Dockerfile.flexible --build-arg COMPUTE_MODE=cpu -t retrieval-api:cpu .

# NVIDIA GPUサポートで実行
docker run -d --name retrieval-api \
  -p 8000:8000 \
  --gpus all \
  retrieval-api

# プロキシ設定で実行
docker run -d --name retrieval-api \
  -p 8000:8000 \
  --gpus all \
  -e HTTP_PROXY=http://proxy.company.com:8080 \
  -e HTTPS_PROXY=http://proxy.company.com:8080 \
  -e NO_PROXY=localhost,127.0.0.1 \
  retrieval-api

# AMD GPUサポートで実行
docker run -d --name retrieval-api-amd \
  -p 8000:8000 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  retrieval-api:amd

# CPU専用で実行
docker run -d --name retrieval-api \
  -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=-1 \
  retrieval-api
```

#### Docker Compose

```bash
# NVIDIA GPUサポート
docker-compose up -d

# プロキシ設定付き
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
export NO_PROXY=localhost,127.0.0.1
docker-compose up -d

# AMD GPUサポート
docker-compose -f docker/docker-compose.amd.yml up -d

# CPU専用モード
docker-compose -f docker/docker-compose.cpu.yml up -d
```

### ローカル開発

```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係インストール
pip install -r requirements.txt

# サービス開始
python run.py
```

## API使用方法

### 利用可能モデル

利用可能なモデルを確認：

```bash
curl http://localhost:8000/models
```

### Rerankエンドポイント

動的モデル選択による文書の再ランキング：

#### デフォルトモデルの使用

```bash
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bce-reranker-base_v1",
    "query": "機械学習とは何ですか？",
    "documents": [
      "機械学習は人工知能の一分野です。",
      "今日は晴れて良い天気です。",
      "深層学習は機械学習の手法の一つです。"
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

#### 日本語モデルの使用

```bash
# 日本語高性能モデルの使用
curl -X POST "http://localhost:8000/v1/rerank" \
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

# 日本語バランス型モデルの使用
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "japanese-reranker-base",
    "query": "自然言語処理の技術について",
    "documents": [
      "NLPはコンピュータが人間の言語を理解するのを助けます。",
      "パスタを茹でるにはまずお湯を沸かします。",
      "テキスト解析はNLPの中核的な要素です。"
    ]
  }'
```

#### 多言語モデルの使用

```bash
# 高性能多言語モデルの使用
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-reranker-v2",
    "query": "持続可能なエネルギーソリューション",
    "documents": [
      "ソーラーパネルは太陽光を電気に変換します",
      "今日は美しい一日です",
      "風力タービンはクリーンエネルギーを生成します",
      "電気自動車は炭素排出量を削減します"
    ],
    "top_n": 3,
    "return_documents": true
  }'

# 高性能大型モデルの使用
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mxbai-rerank-large",
    "query": "artificial intelligence applications",
    "documents": [
      "AI is used in healthcare for diagnosis",
      "The weather is nice today",
      "Machine learning powers recommendation systems",
      "Natural language processing enables chatbots"
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

#### レスポンス例

```json
{
  "model": "jinaai/jina-reranker-v2-base-multilingual",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.9823,
      "document": "ソーラーパネルは太陽光を電気に変換します"
    },
    {
      "index": 2,
      "relevance_score": 0.9156,
      "document": "風力タービンはクリーンエネルギーを生成します"
    }
  ],
  "meta": {
    "api_version": "v1",
    "processing_time_ms": 245,
    "total_documents": 4,
    "returned_documents": 2
  }
}
```

### Embeddings API

#### 埋め込み生成

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": "自然言語処理は魅力的な分野です。"
  }'
```

#### バッチ埋め込み

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": [
      "埋め込む最初のテキスト",
      "埋め込む二番目のテキスト",
      "埋め込む三番目のテキスト"
    ]
  }'
```

#### 日本語モデルの使用

```bash
# 日本語高性能モデルの使用
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ruri-large",
    "input": [
      "自然言語処理は人工知能の重要な分野です。",
      "機械学習アルゴリズムは大量のデータを必要とします。",
      "深層学習は多層ニューラルネットワークを使用します。"
    ]
  }'

# 日本語バランス型モデルの使用
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ruri-base",
    "input": "日本語のテキスト埋め込みを生成します。"
  }'
```

#### 多言語モデルの使用

```bash
# 高性能多言語モデルの使用
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "multilingual-e5-large",
    "input": [
      "東京は日本の首都です",
      "機械学習は人工知能の一部です",
      "自然言語処理は重要です"
    ]
  }'

# 高次元SOTA モデルの使用
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nv-embed-v2",
    "input": "このモデルは4096次元で最先端の埋め込み品質を提供します。"
  }'
```

#### レスポンス例

```json
{
  "object": "list",
  "model": "BAAI/bge-m3",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0234, -0.0156, 0.0789, ...]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [0.0412, -0.0298, 0.0634, ...]
    }
  ],
  "usage": {
    "prompt_tokens": 16,
    "total_tokens": 16
  }
}
```

### その他のエンドポイント

#### ヘルスチェック
```bash
curl http://localhost:8000/health
```

#### モデル一覧
```bash
curl http://localhost:8000/models
```

## API仕様

### POST /v1/rerank

#### リクエストパラメータ

| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| model | string | いいえ | 使用するモデル（短縮名または完全名、デフォルト: "bce-reranker-base_v1"） |
| query | string | はい | 文書をランキングするためのクエリ文字列 |
| documents | array[string] | はい | 再ランキングする文書のリスト（最大1000） |
| top_n | integer | いいえ | 返すトップ結果の数 |
| return_documents | boolean | いいえ | 文書テキストを含めるかどうか（デフォルト: false） |

#### レスポンス

| フィールド | 型 | 説明 |
|-------|------|-------------|
| model | string | 使用されたモデル名 |
| results | array | ランキング結果のリスト |
| results[].index | integer | 元の文書インデックス |
| results[].relevance_score | float | 関連性スコア（0-1） |
| results[].document | string | 文書テキスト（return_documents=trueの場合） |
| meta | object | メタデータ |

### POST /v1/embeddings

#### リクエストパラメータ

| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| model | string | いいえ | 使用するモデル（短縮名または完全名、デフォルト: "bge-m3"） |
| input | string or array[string] | はい | 埋め込むテキスト（最大2048テキスト） |
| encoding_format | string | いいえ | 埋め込みのフォーマット（"float"または"base64"、デフォルト: "float"） |
| dimensions | integer | いいえ | 埋め込みを削減する次元数 |
| user | string | いいえ | ユーザー識別子 |

#### レスポンス

| フィールド | 型 | 説明 |
|-------|------|-------------|
| object | string | 常に "list" |
| model | string | 使用されたモデル名 |
| data | array | 埋め込みオブジェクトのリスト |
| data[].object | string | 常に "embedding" |
| data[].index | integer | 入力テキストのインデックス |
| data[].embedding | array[float] or string | 埋め込みベクトル（floatの配列またはbase64文字列） |
| usage | object | トークン使用情報 |

## モデル管理機能

### プリロードとキャッシング

- **デフォルトモデル**: サービス起動時にプリロードされ、即座に応答
- **オンデマンドローディング**: 最初のリクエスト時にモデルが自動ロード
- **メモリキャッシング**: 後続リクエストのためにモデルをメモリにキャッシュ
- **自動フォールバック**: ロードエラー時にデフォルトモデルにフォールバック

### 動的モデル切り替え

- **APIレベル選択**: 短縮名または完全なモデル名を使用してモデルを切り替え
- **統一管理**: RerankerとEmbeddingモデルの両方で同一の管理パターンを使用
- **エラーハンドリング**: フォールバックメカニズム付きの適切なエラーハンドリング
- **モデル情報**: API経由で詳細なモデルメタデータが利用可能

### 言語サポート

#### 日本語言語最適化
- **専用トークン化**: 日本語固有のテキスト処理
- **高性能**: 日本語コーパスで特別に訓練されたモデル
- **文化的文脈**: 日本語のニュアンスをより良く理解

#### 多言語機能
- **100以上の言語**: 多様な言語処理のサポート
- **クロスリンガル**: 異なる言語間での一貫したパフォーマンス
- **Unicode サポート**: 完全なUnicode文字セットの処理

## 環境変数

| 変数 | デフォルト | 説明 |
|----------|---------|-------------|
| HOST | 0.0.0.0 | サービスホスト |
| PORT | 8000 | サービスポート |
| WORKERS | 1 | ワーカー数 |
| RERANKER_MODEL_NAME | maidalun1020/bce-reranker-base_v1 | デフォルトRerankerモデル名 |
| EMBEDDING_MODEL_NAME | BAAI/bge-m3 | デフォルトEmbeddingモデル名 |
| RERANKER_MODELS_DIR | /app/models | モデル保存のベースディレクトリ |
| HTTP_PROXY | - | HTTPプロキシサーバーURL |
| HTTPS_PROXY | - | HTTPSプロキシサーバーURL |
| NO_PROXY | - | プロキシをバイパスするホストのカンマ区切りリスト |

## 開発

### テスト実行

```bash
# 全テスト実行
pytest tests/

# カバレッジ付き実行
pytest tests/ --cov=.

# 特定のテスト実行
python -m pytest tests/test_api.py -v

# API例テスト実行
python tests/test_api_example.py

# ハードウェア検出テスト実行
bash tests/test_detection.sh
```

### Dockerテスト

```bash
# 異なるDocker設定をテスト
docker-compose up -d                                      # NVIDIA GPU（ルート）
docker-compose -f docker/docker-compose.yml up -d         # NVIDIA GPU（docker/）
docker-compose -f docker/docker-compose.amd.yml up -d     # AMD GPU
docker-compose -f docker/docker-compose.cpu.yml up -d     # CPU専用

# 特定のDockerファイルでテスト
docker build -f docker/Dockerfile.amd -t test:amd .
docker build -f docker/Dockerfile.flexible --build-arg COMPUTE_MODE=cpu -t test:cpu .
```

### 手動APIテスト

付属のテストスクリプトを使用：

```bash
python tests/test_api_example.py
```

## Docker設定

### ビルド引数

```bash
# プロキシサポート付きビルド
docker build \
  --build-arg HTTP_PROXY=http://proxy.company.com:8080 \
  --build-arg HTTPS_PROXY=http://proxy.company.com:8080 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  -t retrieval-api .

# AMD GPUバージョンビルド
docker build -f docker/Dockerfile.amd -t retrieval-api:amd .

# CPU専用バージョンビルド
docker build -f docker/Dockerfile.flexible \
  --build-arg COMPUTE_MODE=cpu \
  -t retrieval-api:cpu .
```

### GPUサポート

GPUサポートには以下が必要です：

1. NVIDIAドライバーがインストール済み
2. NVIDIA Container Toolkitがインストール済み
3. GPU アクセス用にDockerが設定済み

```bash
# GPU アクセステスト
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## モデル管理

### モデルキャッシング

モデルは最初のロード後に自動的にキャッシュされます。キャッシュディレクトリ構造：

```
/app/models/
├── rerankers/
│   ├── maidalun1020_bce-reranker-base_v1/
│   ├── jinaai_jina-reranker-v2-base-multilingual/
│   └── hotchpotch_japanese-reranker-cross-encoder-large-v1/
└── embeddings/
    ├── BAAI_bge-m3/
    ├── cl-nagoya_ruri-large/
    └── intfloat_multilingual-e5-large/
```

### Dockerファイル構造

プロジェクトには複数のDocker設定が含まれています：

```
docker/
├── Dockerfile                  # 標準NVIDIA GPU ビルド
├── Dockerfile.amd             # AMD ROCm GPU サポート
├── Dockerfile.flexible        # CPU/GPU フレキシブルビルド
├── docker-compose.yml         # 標準composeファイル
├── docker-compose.amd.yml     # AMD GPU compose
├── docker-compose.cpu.yml     # CPU専用compose
├── requirements.txt           # 標準要件
├── requirements.amd.txt       # AMD固有要件
└── requirements-cpu.txt       # CPU専用要件
```

**注記**: 便宜上、メインの`docker-compose.yml`もルートディレクトリで利用可能です。

### カスタムモデル

カスタムモデルを追加するには、`reranker_loader.py`または`embedding_loader.py`の`supported_models`辞書を更新してください：

```python
self.supported_models = {
    "your-custom/model-name": {
        "name": "custom-model",
        "description": "あなたのカスタムモデル",
        "max_length": 512
    }
}
```

## パフォーマンス最適化

### GPU設定

#### NVIDIA GPU サポート
- NVIDIAドライバー（最新バージョン推奨）
- CUDA 11.8+サポート
- GPU メモリ 4GB+推奨
- Docker用NVIDIA Container Toolkit

#### AMD GPU サポート
- ROCm 6.0+サポート
- AMD GPU ドライバー（AMDGPU-PROまたはオープンソース）
- GPU メモリ 4GB+推奨
- AMD GPU デバイスアクセス付きDocker（`/dev/kfd`、`/dev/dri`）

#### 自動検出
サービスは利用可能なGPUハードウェアを自動検出します：
- 🟢 NVIDIA GPU → CUDA アクセラレーション使用
- 🔵 AMD GPU → ROCm アクセラレーション使用
- ⚪ GPU なし → CPU にフォールバック

### メモリ管理

- **効率的キャッシング**: 最初のロード後にモデルがキャッシュされ、後続リクエストが高速化
- **バッチ処理**: 複数の文書/テキストがまとめて処理され、スループットが向上
- **メモリ監視**: 自動メモリクリーンアップと監視
- **リソース制限**: Docker デプロイメント用の設定可能なメモリ制限

## トラブルシューティング

### GPU が検出されない

エラー: `could not select device driver "nvidia" with capabilities: [[gpu]]`が発生した場合

1. CPU専用モードを使用：
```bash
# docker-composeを使用
docker-compose -f docker/docker-compose.cpu.yml up -d

# docker runを使用
docker run -d --name retrieval-api \
  -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=-1 \
  retrieval-api

# または自動開始スクリプトを使用
./start.sh
```

2. GPUサポートを修正するには、以下を確認：
```bash
# NVIDIAドライバー確認
nvidia-smi

# NVIDIA Container Toolkitインストール
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Docker GPUサポート検証
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 遅いモデルダウンロード

Hugging Face ミラーを使用：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### メモリ問題

Docker デプロイメントでメモリ制限を増加：
```yaml
deploy:
  resources:
    limits:
      memory: 8G
```

### モデルロードエラー

1. ディスク容量を確認
2. Hugging Face Hubへのネットワーク接続を確認
3. モデル名のスペルを確認
4. 詳細なエラーメッセージのためのログを確認

## API例

### Pythonクライアント例

```python
import requests
import numpy as np

# Reranking例
def rerank_documents(query, documents, model="jina-reranker-v2"):
    response = requests.post("http://localhost:8000/v1/rerank", json={
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": 5,
        "return_documents": True
    })
    return response.json()

# Embedding例
def create_embeddings(texts, model="bge-m3"):
    response = requests.post("http://localhost:8000/v1/embeddings", json={
        "model": model,
        "input": texts
    })
    return response.json()

# 日本語モデルでの使用例
japanese_query = "人工知能の応用分野について"
japanese_docs = [
    "AIは医療診断で重要な役割を果たしています",
    "今日は良い天気です",
    "機械学習は推薦システムに使われています",
    "自然言語処理はチャットボットを可能にします"
]

# 日本語モデルでRerank
rerank_results = rerank_documents(
    japanese_query, 
    japanese_docs, 
    model="japanese-reranker-large"
)

for result in rerank_results["results"]:
    print(f"スコア: {result['relevance_score']:.3f} - {result['document']}")

# 日本語モデルで埋め込み生成
embed_results = create_embeddings(
    ["東京は日本の首都です", "機械学習は人工知能の分野です"],
    model="ruri-large"
)

embeddings = [item['embedding'] for item in embed_results['data']]
print(f"{len(embeddings)}個の埋め込みを{len(embeddings[0])}次元で生成しました")
```

### JavaScript/Node.js例

```javascript
const axios = require('axios');

async function rerankDocuments(query, documents, model = 'jina-reranker-v2') {
  try {
    const response = await axios.post('http://localhost:8000/v1/rerank', {
      model,
      query,
      documents,
      top_n: 5,
      return_documents: true
    });
    return response.data;
  } catch (error) {
    console.error('エラー:', error.response?.data || error.message);
    throw error;
  }
}

async function createEmbeddings(input, model = 'bge-m3') {
  try {
    const response = await axios.post('http://localhost:8000/v1/embeddings', {
      model,
      input
    });
    return response.data;
  } catch (error) {
    console.error('エラー:', error.response?.data || error.message);
    throw error;
  }
}

// 使用例
const query = "持続可能なエネルギーソリューション";
const docs = [
  "ソーラーパネルは太陽光を電気に変換します",
  "今日は美しい一日です",
  "風力タービンはクリーンエネルギーを生成します",
  "電気自動車は炭素排出量を削減します"
];

// 文書をRerank
rerankDocuments(query, docs, 'mxbai-rerank-large').then(results => {
  console.log('再ランキング結果:');
  results.results.forEach((result, index) => {
    console.log(`${index + 1}. スコア: ${result.relevance_score.toFixed(3)} - ${result.document}`);
  });
});

// 埋め込み生成
createEmbeddings([
  "人工知能は産業を変革しています",
  "深層学習モデルは大規模なデータセットを必要とします"
], 'multilingual-e5-large').then(results => {
  console.log(`${results.data.length}個の埋め込みを生成しました`);
  console.log(`埋め込み次元: ${results.data[0].embedding.length}`);
});
```

## ライセンス

このプロジェクトはMITライセンスの下でリリースされています。

## 貢献

プルリクエストは歓迎します。大きな変更の場合は、まず問題を開いて提案された変更について議論してください。

## サポート

問題が発生した場合は、GitHub Issuesページで報告してください。

---

**注記**: このサービスは文書再ランキングとテキスト埋め込み機能を提供し、適切な監視とスケーリングの考慮事項を備えた本番使用向けに設計されています。