# 🏺 Vesuvius Challenge - 3D Surface Detection

実データ対応のVesuvius Challenge（ベスヴィオ火山巻物）3D表面検出PyTorchパイプライン。

## ✨ 主な特徴

- ✅ **実Kaggleデータ自動検出・ロード**
- ✅ **実データなしでも高品質デモデータで動作**  
- ✅ **PyTorch 3D CNN（ResNet3D, UNet3D, SwinUNetr）**
- ✅ **完全な訓練・検証・推論パイプライン**
- ✅ **整理されたディレクトリ構造**

## 📁 プロジェクト構造

```
vesuvius-challenge-surface-detection/
│
├── 📂 notebooks/              # Jupyterノートブック
│   ├── 📂 training/          # 学習用ノートブック
│   │   ├── main_training.ipynb      # メイン学習（推奨）
│   │   ├── swinunetr_training.ipynb # SwinUNetr学習
│   │   └── swinunetr_v2.ipynb       # SwinUNetr v2
│   │
│   ├── 📂 inference/         # 推論用ノートブック
│   │   └── inference.ipynb   # 推論・予測・提出
│   │
│   └── 📂 runpods/           # Runpods環境用
│       ├── runpods_complete.ipynb
│       ├── runpods_training.ipynb
│       └── runpods_standalone.ipynb
│
├── 📂 src/                    # ソースコード
│   ├── download_kaggle_data.py  # Kaggleデータ自動取得
│   └── unified_data_loader.py   # 統合データローダー
│
├── 📂 docs/                   # ドキュメント
│   ├── REAL_DATA_SETUP.md    # データセットアップ
│   ├── FILE_STRUCTURE.md     # ファイル構造説明
│   └── upload_to_runpods.md  # Runpods設定
│
├── 📂 scripts/                # スクリプト
│   └── runpods_safe_setup.sh # 環境セットアップ
│
├── README.md                  # このファイル
├── requirements.txt           # 必要パッケージ
└── .gitignore                # Git除外設定
```

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# 依存関係インストール
pip install -r requirements.txt

# Runpods環境の場合
bash scripts/runpods_safe_setup.sh
```

### 2. メイン学習（推奨）

```bash
# Jupyter起動
jupyter notebook notebooks/training/main_training.ipynb
```

このノートブックで全て実行可能：
- 実データ自動検出
- 3D CNN学習
- 結果可視化
- Kaggle提出

### 3. 高性能モデル（オプション）

```bash
# SwinUNetr使用
jupyter notebook notebooks/training/swinunetr_training.ipynb
```

## 📊 データ設定

### 実データ配置

Kaggleデータを以下の構造で配置：

```
data/
└── vesuvius-challenge-surface-detection/
    ├── train_images/
    │   └── *.tif
    ├── train_labels/  (オプション)
    │   └── *.tif
    └── train.csv
```

詳細は `docs/REAL_DATA_SETUP.md` を参照。

### デモデータモード

実データが見つからない場合、自動的に高品質デモデータで動作します。

## 🎓 使用手順

### ステップ 1: データ準備
```python
# notebooks/training/main_training.ipynb で自動実行
from src.unified_data_loader import UnifiedVesuviusDataLoader

loader = UnifiedVesuviusDataLoader()
info = loader.get_data_info()
```

### ステップ 2: 学習実行
```python
# notebooks/training/main_training.ipynb のセルを順次実行
# 設定はconfig辞書で調整可能
config = {
    'batch_size': 4,
    'num_epochs': 20,
    'model_type': 'unet3d'  # or 'resnet3d'
}
```

### ステップ 3: 推論・提出
```bash
jupyter notebook notebooks/inference/inference.ipynb
```

## 🏗️ モデルアーキテクチャ

### 利用可能モデル
- **ResNet3D**: 軽量3D CNN
- **UNet3D**: U-Netベース3Dセグメンテーション  
- **SwinUNetr**: Swin Transformer + U-Net（最高性能）

### 選択基準
- **ResNet3D**: GPU制限環境
- **UNet3D**: バランス型（推奨）
- **SwinUNetr**: 最高精度追求

## 📈 性能最適化

### GPU不足の場合
```python
config = {
    'batch_size': 2,         # 削減
    'volume_size': (64, 64),  # 縮小
    'volume_depth': 8         # 削減
}
```

### 高速化
- `num_workers=4` でデータローディング並列化
- Mixed precision学習対応
- Gradient accumulation利用可能

## 🛠️ トラブルシューティング

### よくある問題

1. **CUDA out of memory**
   - バッチサイズを削減
   - ボリュームサイズを縮小

2. **データが見つからない**
   - `docs/REAL_DATA_SETUP.md` 確認
   - デモデータモードで継続

3. **学習が収束しない**
   - 学習率を調整
   - データ拡張を追加

## 📚 ドキュメント

- `docs/REAL_DATA_SETUP.md` - データセットアップ詳細
- `docs/FILE_STRUCTURE.md` - プロジェクト構造説明
- `docs/upload_to_runpods.md` - クラウド環境設定

## 📄 ライセンス

Vesuvius Challenge公式ルールに準拠。

## 🎯 貢献

改善提案・バグ報告はIssueでお願いします。

---

**開始方法**: `jupyter notebook notebooks/training/main_training.ipynb` を実行！