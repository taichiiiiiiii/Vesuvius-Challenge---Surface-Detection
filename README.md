# 🏺 Vesuvius Challenge - 3D Surface Detection

古代ヴェスヴィオ火山の噴火で埋もれた巻物の3D CTスキャンから、インクの痕跡を検出するディープラーニングプロジェクト。

## 📌 概要

このプロジェクトは、Kaggle Vesuvius Challengeのための3D表面検出システムです。PyTorchベースの3D CNNを使用して、巻物のCTスキャン画像からインクパターンを識別します。

## ✨ 主な特徴

- 🔍 **自動データ検出** - Kaggleデータを自動的に検出・ロード
- 🎯 **複数の3Dモデル** - ResNet3D、UNet3D、SwinUNetrをサポート
- 📊 **完全なMLパイプライン** - データ前処理から学習、推論、提出まで
- ☁️ **Runpods対応** - クラウドGPU環境での実行に最適化
- 🚀 **すぐに実行可能** - 最小限の設定で動作

## 📁 プロジェクト構成

```
.
├── notebooks/
│   ├── training/              # 学習用ノートブック
│   │   ├── main_training.ipynb      # メイン学習スクリプト ⭐
│   │   ├── swinunetr_training.ipynb # 高性能モデル学習
│   │   └── swinunetr_v2.ipynb       # 改良版モデル
│   │
│   ├── inference/             # 推論・予測用
│   │   └── inference.ipynb   # 提出ファイル生成
│   │
│   └── runpods/              # クラウド環境用
│       └── runpods_complete.ipynb   # Runpods完全版
│
├── src/                      # ソースコード
│   ├── unified_data_loader.py       # 統合データローダー
│   └── download_kaggle_data.py      # データ自動取得
│
├── docs/                     # ドキュメント
│   ├── REAL_DATA_SETUP.md   # データ準備ガイド
│   └── upload_to_runpods.md # Runpods設定方法
│
└── scripts/                  # セットアップスクリプト
    └── runpods_safe_setup.sh # 環境構築スクリプト
```

## 🚀 クイックスタート

### 前提条件

- Python 3.8以上
- CUDA対応GPU（推奨: 8GB以上のVRAM）
- 50GB以上のディスクスペース

### 1. 環境構築

```bash
# リポジトリをクローン
git clone https://github.com/taichiiiiiiii/Vesuvius-Challenge---Surface-Detection.git
cd Vesuvius-Challenge---Surface-Detection

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. データ準備

#### オプション A: Kaggleから自動ダウンロード

```python
# notebooks/training/main_training.ipynb 内で実行
from src.download_kaggle_data import download_competition_data
download_competition_data()
```

#### オプション B: 手動配置

```bash
# データを以下の構造で配置
data/
└── vesuvius-challenge-surface-detection/
    ├── train_images/      # 訓練画像
    ├── train_labels/      # ラベル（オプション）
    └── train.csv          # メタデータ
```

### 3. 学習実行

```bash
# Jupyterノートブックを起動
jupyter notebook notebooks/training/main_training.ipynb
```

ノートブックを開き、セルを順番に実行してください。

### 4. 推論・提出

```bash
# 学習済みモデルで予測
jupyter notebook notebooks/inference/inference.ipynb
```

## 🏗️ モデルアーキテクチャ

| モデル | 特徴 | 推奨用途 |
|--------|------|----------|
| **UNet3D** | バランス型 | 一般的な使用（推奨） |
| **ResNet3D** | 軽量・高速 | メモリ制限環境 |
| **SwinUNetr** | 最高精度 | 高性能GPU環境 |

## ⚙️ 設定カスタマイズ

```python
# notebooks/training/main_training.ipynb で設定
config = {
    'model_type': 'unet3d',     # モデル選択
    'batch_size': 4,             # バッチサイズ
    'num_epochs': 20,            # エポック数
    'learning_rate': 1e-4,       # 学習率
    'volume_size': (128, 128),   # ボリュームサイズ
    'volume_depth': 16           # Z軸の深さ
}
```

## 💡 メモリ最適化

GPU メモリが不足する場合：

```python
# 設定を調整
config = {
    'batch_size': 2,             # 小さく
    'volume_size': (64, 64),     # 縮小
    'volume_depth': 8,           # 浅く
    'gradient_accumulation': 4   # 勾配累積を使用
}
```

## 📊 パフォーマンス

| 設定 | VRAM使用量 | 学習時間/エポック | 推奨GPU |
|------|------------|-------------------|---------|
| フル | ~12GB | ~15分 | RTX 3090/4090 |
| 標準 | ~8GB | ~10分 | RTX 3070/3080 |
| 軽量 | ~4GB | ~5分 | RTX 3060/2070 |

## 🛠️ トラブルシューティング

### CUDA out of memory
- バッチサイズを半分に削減
- ボリュームサイズを64x64に縮小
- Mixed Precision学習を有効化

### データが見つからない
- Kaggle APIトークンを設定（`~/.kaggle/kaggle.json`）
- `docs/REAL_DATA_SETUP.md`を参照
- デモデータモードで動作確認

### 学習が収束しない
- 学習率を1/10に削減
- データ拡張を調整
- より長いエポック数で学習

## 📚 詳細ドキュメント

- [データ準備ガイド](docs/REAL_DATA_SETUP.md)
- [ファイル構造説明](docs/FILE_STRUCTURE.md)
- [Runpods環境セットアップ](docs/upload_to_runpods.md)

## 🤝 貢献

プルリクエスト歓迎です！大きな変更の場合は、まずIssueを作成して変更内容を議論してください。

## 📄 ライセンス

このプロジェクトは[Vesuvius Challenge](https://scrollprize.org/)の公式ルールに準拠しています。

## 🙏 謝辞

- Vesuvius Challengeの主催者とコミュニティ
- PyTorch、MONAI開発チーム
- Kaggleプラットフォーム

---

**📮 質問・サポート**: GitHubのIssuesページをご利用ください。

**⭐ このプロジェクトが役立った場合は、スターをお願いします！**