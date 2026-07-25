# Vesuvius Challenge 実データセットアップ手順

## 概要

- データローダー: `src/unified_data_loader.py`（実データ自動検出・デモデータフォールバック付き）
- ダウンローダー: `src/download_kaggle_data.py`（Kaggle API経由の自動取得）
- 実データが見つからない場合はデモデータで学習パイプラインの動作確認が可能

## 1. Kaggle認証の設定

> ⚠️ **セキュリティ**: `kaggle.json` は**リポジトリ内に置かないでください**。
> 推奨は環境変数、または `~/.kaggle/kaggle.json`（Runpodsでは `/workspace/kaggle.json`）です。
> 詳細は [SECURITY.md](../SECURITY.md) を参照。

```bash
# 方法A: 環境変数（推奨）
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# 方法B: 標準の場所に配置
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

APIトークンは [Kaggle Settings](https://www.kaggle.com/settings/account) → "Create New API Token" で取得できます。

## 2. データの自動ダウンロード

```bash
python src/download_kaggle_data.py
```

または Python から:

```python
from src.download_kaggle_data import download_vesuvius_dataset
dataset_path = download_vesuvius_dataset(output_dir="./data")
```

## 3. 手動配置する場合のディレクトリ構造

以下のいずれかの場所に配置してください（自動検出されます）:

```
./data/vesuvius-challenge-surface-detection/     # ローカル推奨
/workspace/vesuvius-challenge-surface-detection/ # Runpods
./vesuvius-challenge-surface-detection/

└── vesuvius-challenge-surface-detection/
    ├── train_images/      # 3D CTボリューム (*.tif)
    ├── train_labels/      # セグメンテーションマスク (*.tif, オプション)
    └── train.csv          # メタデータ
```

## 4. 動作確認

```bash
python src/unified_data_loader.py
```

実データが見つかった場合の出力例:

```
✅ 実データ自動検出: ./data/vesuvius-challenge-surface-detection
📊 150個のTIFFファイル発見
✅ 実データ使用: ./data/vesuvius-challenge-surface-detection
```

## 5. 学習での使用

```python
from src.unified_data_loader import create_data_loaders

train_loader, val_loader = create_data_loaders(
    volume_size=(96, 96, 64),
    batch_size=4,
    data_path=None,   # None = 自動検出
)
```

## フォールバック動作

実データが見つからない場合:

- ✅ 高品質デモデータを自動生成
- ✅ 同一APIで動作継続（コード変更不要）
- ✅ パイプラインの動作確認・デバッグに利用可能

## 注意事項

1. **容量**: 生データは約25GB。ディスク空き容量を確認してください
2. **メモリ**: ローダーはスライス単位で読み込み、メモリ効率を優先しています
3. **ラベルなしデータ**: `train_labels/` がない場合は閾値ベースの簡易ラベルを生成します
