# Rerank API Service

🇯🇵 日本語 | [🇺🇸 English](README.md)

OpenAI互換のRerank APIサービス。BGE Rerankerモデルを使用して高精度な文書の再ランキング機能を提供します。

## 機能

- OpenAI API互換のRerankエンドポイント
- APIリクエストによる動的なモデル選択
- BGE Rerankerモデルによる高精度な文書再ランキング
- マルチGPUサポート（NVIDIA CUDA、AMD ROCm）と自動検出
- CPUフォールバックサポート
- Dockerによる簡単なデプロイメント
- 非同期処理による高速レスポンス
- モデルキャッシュと効率的なメモリ管理

## クイックスタート

### Dockerを使用した起動

#### 自動GPU/CPU検出

提供されているスクリプトで自動検出：

```bash
# スクリプトに実行権限を付与
chmod +x start.sh

# GPU/CPUを自動検出して起動
./start.sh
```

#### 手動Dockerコマンド

```bash
# ビルド
docker build -t rerank-api .

# GPU使用（利用可能な場合）
docker run -d --name rerank-api \
  -p 7987:7987 \
  --gpus all \
  rerank-api

# CPU専用モード
docker run -d --name rerank-api \
  -p 7987:7987 \
  -e CUDA_VISIBLE_DEVICES=-1 \
  rerank-api
```

#### Docker Compose

```bash
# GPU使用
docker-compose up -d

# CPU専用モード
docker-compose -f docker-compose.cpu.yml up -d
```

### ローカル環境での起動

```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 依存関係インストール
pip install -r requirements.txt

# サービス起動
python run.py
```

## API使用方法

### 対応モデル

このAPIは複数のリランキングモデルに対応しています：

| モデル名 | 短縮名 | 説明 |
|---------|-------|------|
| maidalun1020/bce-reranker-base_v1 | bce-reranker-base_v1 | BGE Reranker Base Model v1（デフォルト） |
| BAAI/bge-reranker-base | bge-reranker-base | BGE Reranker Base Model |
| BAAI/bge-reranker-large | bge-reranker-large | BGE Reranker Large Model |

### Rerankエンドポイント

文書を再ランキングします。モデルはAPIリクエストで動的に指定できます：

#### デフォルトモデル使用例
```bash
curl -X POST "http://localhost:7987/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bce-reranker-base_v1",
    "query": "機械学習とは何ですか？",
    "documents": [
      "機械学習は人工知能の一分野です。",
      "今日は良い天気です。",
      "ディープラーニングは機械学習の手法の一つです。"
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

#### 異なるモデル使用例
```bash
curl -X POST "http://localhost:7987/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-large",
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a branch of artificial intelligence.",
      "The weather is nice today.",
      "Deep learning is a method of machine learning."
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

レスポンス例：
```json
{
  "model": "bce-reranker-base_v1",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": "機械学習は人工知能の一分野です。"
    },
    {
      "index": 2,
      "relevance_score": 0.87,
      "document": "ディープラーニングは機械学習の手法の一つです。"
    }
  ],
  "meta": {
    "api_version": "v1",
    "processing_time_ms": 145,
    "total_documents": 3,
    "returned_documents": 2
  }
}
```

### その他のエンドポイント

#### ヘルスチェック
```bash
curl http://localhost:7987/health
```

#### モデル一覧
```bash
curl http://localhost:7987/models
```

## API仕様

### POST /v1/rerank

#### リクエストパラメータ

| パラメータ | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| model | string | いいえ | 使用するモデル（短縮名または完全名、デフォルト: "bce-reranker-base_v1"） |
| query | string | はい | ランキングの基準となるクエリ文字列 |
| documents | array[string] | はい | ランキング対象の文書リスト（最大1000件） |
| top_n | integer | いいえ | 返却する上位結果数 |
| return_documents | boolean | いいえ | 文書テキストを含めるか（デフォルト: false） |

#### レスポンス

| フィールド | 型 | 説明 |
|-----------|-----|------|
| model | string | 使用されたモデル名 |
| results | array | ランキング結果のリスト |
| results[].index | integer | 元の文書リストでのインデックス |
| results[].relevance_score | float | 関連性スコア（0-1） |
| results[].document | string | 文書テキスト（return_documents=trueの場合） |
| meta | object | メタデータ |

## 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| HOST | 0.0.0.0 | サービスホスト |
| PORT | 7987 | サービスポート |
| WORKERS | 1 | ワーカー数 |
| RERANKER_MODEL_NAME | maidalun1020/bce-reranker-base_v1 | デフォルトモデル名 |
| RERANKER_MODELS_DIR | /app/models | モデル保存ベースディレクトリ |

## 開発

### テスト実行

```bash
# API動作テスト
pytest tests/

# 個別のテスト
python -m pytest tests/test_api.py -v
```

### コード品質チェック

```bash
# フォーマット
black .

# Linting
ruff check .

# 型チェック
mypy app.py
```

## トラブルシューティング

### GPU が認識されない場合

エラー `could not select device driver "nvidia" with capabilities: [[gpu]]` が発生した場合：

1. CPU専用モードを使用する：
```bash
# Docker Composeを使用
docker-compose -f docker-compose.cpu.yml up -d

# Docker runを使用
docker run -d --name rerank-api \
  -p 7987:7987 \
  -e CUDA_VISIBLE_DEVICES=-1 \
  rerank-api

# または自動起動スクリプトを使用
./start.sh
```

2. GPUサポートを修正するには：
```bash
# NVIDIAドライバーの確認
nvidia-smi

# NVIDIA Container Toolkitのインストール
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Docker GPUサポートの確認
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### モデルのダウンロードが遅い場合

Hugging Face のミラーを使用：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### メモリ不足エラー

Docker の場合、メモリ制限を増やす：
```yaml
deploy:
  resources:
    limits:
      memory: 8G
```

### モデル読み込みエラー

1. ディスク容量の確認
2. Hugging Face Hubへのネットワーク接続確認
3. モデル名のスペル確認
4. 詳細なエラーメッセージのログ確認

## モデル管理

### モデルキャッシュ

モデルは最初の読み込み後に自動的にキャッシュされます。キャッシュディレクトリ構造：

```
/app/models/
├── maidalun1020_bce-reranker-base_v1/
├── BAAI_bge-reranker-base/
└── BAAI_bge-reranker-large/
```

### カスタムモデル

カスタムモデルを追加するには、`model_loader.py`の`supported_models`辞書を更新してください：

```python
self.supported_models = {
    "your-custom/model-name": {
        "name": "custom-model",
        "description": "Your Custom Model",
        "max_length": 512
    }
}
```

## API使用例

### Pythonクライアント例

```python
import requests

def rerank_documents(query, documents, model="bce-reranker-base_v1"):
    response = requests.post("http://localhost:7987/v1/rerank", json={
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": 5,
        "return_documents": True
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

results = rerank_documents(query, docs)
for result in results["results"]:
    print(f"スコア: {result['relevance_score']:.3f} - {result['document']}")
```

### JavaScript/Node.js例

```javascript
const axios = require('axios');

async function rerankDocuments(query, documents, model = 'bce-reranker-base_v1') {
  try {
    const response = await axios.post('http://localhost:7987/v1/rerank', {
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

// 使用例
const query = "機械学習アルゴリズム";
const docs = [
  "機械学習は人工知能の一分野です",
  "今日の天気は晴れで暖かいです", 
  "ニューラルネットワークは強力なMLアルゴリズムです",
  "料理には新鮮な食材が必要です"
];

rerankDocuments(query, docs).then(results => {
  results.results.forEach(result => {
    console.log(`スコア: ${result.relevance_score.toFixed(3)} - ${result.document}`);
  });
});
```

### APIテストスクリプト

付属のテストスクリプトを使用：

```bash
python test_api_example.py
```

## パフォーマンス最適化

### GPU設定

- NVIDIAドライバー（最新版推奨）
- CUDA 11.8以上対応
- GPU memory 4GB以上推奨

### メモリ管理

- 効率的なモデルキャッシュ
- バッチ処理によるスループット向上
- 設定可能なワーカー数
- 自動メモリクリーンアップ

## ライセンス

このプロジェクトは MIT ライセンスのもとで公開されています。

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まず Issue を作成して変更内容を議論してください。

## サポート

問題が発生した場合は、GitHub の Issue ページで報告してください。

---

**注意**: このサービスは文書再ランキング機能を提供し、適切な監視とスケーリングを考慮した本番環境での使用を想定して設計されています。