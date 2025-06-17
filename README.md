# Rerank API Service

OpenAI互換のRerank APIサービス。BGE Rerankerモデル（bce-reranker-base_v1）を使用して文書の再ランキング機能を提供します。

## 機能

- OpenAI API互換のRerankエンドポイント
- BGE Rerankerモデルによる高精度な文書再ランキング
- GPU/CPU自動検出とサポート
- Dockerによる簡単なデプロイメント
- 非同期処理による高速レスポンス

## クイックスタート

### Dockerを使用した起動

```bash
# ビルド
docker build -t rerank-api .

# 起動（GPU使用）
docker run -d --name rerank-api \
  -p 7987:7987 \
  --gpus all \
  rerank-api

# または Docker Compose を使用
docker-compose up -d
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

### Rerankエンドポイント

文書を再ランキングします：

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
| model | string | いいえ | 使用するモデル（デフォルト: "bce-reranker-base_v1"） |
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
| RERANKER_MODEL_NAME | maidalun1020/bce-reranker-base_v1 | モデル名 |
| RERANKER_MODEL_DIR | /app/models/bce-reranker-base_v1 | モデル保存ディレクトリ |

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

1. NVIDIA ドライバーの確認：
```bash
nvidia-smi
```

2. Docker で GPU を使用する場合：
```bash
# NVIDIA Container Toolkit のインストール確認
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

## ライセンス

このプロジェクトは MIT ライセンスのもとで公開されています。

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まず Issue を作成して変更内容を議論してください。

## サポート

問題が発生した場合は、GitHub の Issue ページで報告してください。