# 🏺 Vesuvius Challenge - 3D Surface Detection

古代ヴェスヴィオ火山の噴火で埋もれた巻物の3D CTスキャンから、パピルス表面のインク痕跡を検出するディープラーニングプロジェクト。

[Kaggle Vesuvius Challenge - Surface Detection](https://scrollprize.org/) 向けの3Dセマンティックセグメンテーション実装です。

## 📌 概要

このリポジトリには2系統の実装があります。

| 実装 | 場所 | 状態 |
|------|------|------|
| **nnU-Net v2**（ResNetエンコーダー付きU-Net） | `notebooks/nnunet/` | ⭐ **現行・推奨** |
| PyTorch 3D CNN（UNet3D / ResNet3D / SwinUNETR） | `notebooks/training/`, `notebooks/runpods/` | 旧実装（参考用） |

### 主な特徴

- 🏆 **nnU-Net v2** - 医療画像セグメンテーションの業界標準フレームワークによる学習パイプライン
- 🔍 **自動データ検出** - Kaggleデータの自動ダウンロード・検出・ロード
- ☁️ **Runpods対応** - クラウドGPU環境での実行に最適化（GPU種別の自動検出）
- 🎭 **デモデータフォールバック** - 実データなしでもパイプラインの動作確認が可能

## 📁 プロジェクト構成

```
.
├── README.md                  # このファイル
├── SECURITY.md                # 🔒 セキュリティポリシー（必読）
├── requirements.txt           # 依存パッケージ
│
├── notebooks/
│   ├── nnunet/                # ⭐ 現行実装（nnU-Net v2）
│   │   └── vesuvius_nnunet_runpods.ipynb
│   ├── training/              # 旧実装: ローカル/汎用 学習ノートブック
│   ├── inference/             # 推論・提出ファイル生成
│   └── runpods/               # 旧実装: Runpods向けオールインワン版
│
├── src/                       # 共通モジュール
│   ├── unified_data_loader.py       # 統合データローダー
│   └── download_kaggle_data.py      # Kaggleデータ自動取得
│
├── scripts/                   # セットアップ・修正スクリプト
│   ├── runpods_safe_setup.sh        # Runpods環境構築
│   ├── runpods_fix_nnunet.sh        # nnU-Netエラー一括修正
│   ├── convert_tiff_to_nifti.py     # TIFF→NIfTI変換
│   ├── fix_nnunet_cv_error.py       # Cross-validationエラー修正
│   └── fix_nnunet_io_error.py       # SimpleTiffIOエラー修正
│
└── docs/                      # 詳細ドキュメント
    ├── FILE_STRUCTURE.md      # ファイル構造の説明
    ├── REAL_DATA_SETUP.md     # データ準備ガイド
    └── upload_to_runpods.md   # Runpodsへのファイル転送方法
```

## 🔒 セキュリティ（最初に読んでください）

- **`kaggle.json`（APIキー）を絶対にコミットしないでください。** リポジトリ外（`~/.kaggle/` または Runpodsの `/workspace/`）に配置するか、環境変数 `KAGGLE_USERNAME` / `KAGGLE_KEY` を使用します
- ノートブックのセルに認証情報を書いたまま保存・共有しないでください
- コミット前に出力セルをクリアしてください: `jupyter nbconvert --clear-output --inplace <notebook>.ipynb`

詳細は **[SECURITY.md](SECURITY.md)** を参照してください。

## 🚀 クイックスタート

### 前提条件

- Python 3.9以上
- CUDA対応GPU（nnU-Net v2はVRAM 16GB以上を推奨）
- ディスク空き容量 50GB以上（生データ約25GB）

### 1. 環境構築

```bash
git clone https://github.com/taichiiiiiiii/Vesuvius-Challenge---Surface-Detection.git
cd Vesuvius-Challenge---Surface-Detection
pip install -r requirements.txt
```

### 2. Kaggle認証の設定

```bash
# 推奨: 環境変数
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# または標準の場所に配置（リポジトリ内には置かない）
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 3. データ準備

```bash
python src/download_kaggle_data.py
```

または手動配置（詳細は [docs/REAL_DATA_SETUP.md](docs/REAL_DATA_SETUP.md)）:

```
data/vesuvius-challenge-surface-detection/
├── train_images/      # 3D CTボリューム (*.tif)
├── train_labels/      # セグメンテーションマスク (*.tif)
└── train.csv          # メタデータ
```

### 4a. 学習（現行: nnU-Net v2 / Runpods推奨）

```bash
# Runpods等のGPU環境で
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
# → notebooks/nnunet/vesuvius_nnunet_runpods.ipynb を開いて上から実行
```

GPU別の推奨設定:

| GPU | VRAM | Configuration | パッチサイズ | 250エポック推定時間 |
|-----|------|---------------|-------------|-------------------|
| A6000 | 48GB | 3d_fullres | (96,96,96) | 12-15時間 |
| RTX 4090 | 24GB | 3d_fullres | (80,80,80) | 15-20時間 |
| T4 | 16GB | 3d_lowres | (64,64,64) | 20-25時間 |

### 4b. 学習（旧実装: PyTorch 3D CNN / ローカル向け）

```bash
jupyter notebook notebooks/training/main_training.ipynb
```

### 5. 推論・提出

```bash
# nnU-Net v2
nnUNetv2_predict -d 100 -c 3d_fullres -f all \
    -i /path/to/test_images -o /path/to/predictions \
    -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_250epochs

# 旧実装
jupyter notebook notebooks/inference/inference.ipynb
```

## 🛠️ トラブルシューティング

### CUDA out of memory
- nnU-Net: configurationを `3d_lowres` に変更
- 旧実装: バッチサイズを半分に、ボリュームサイズを縮小

### nnU-Netのエラー（SimpleTiffIO / Cross-validation）
```bash
bash scripts/runpods_fix_nnunet.sh          # 一括修正
python scripts/fix_nnunet_io_error.py       # TIFF I/Oエラー
python scripts/fix_nnunet_cv_error.py       # n_splits エラー
```

### データが見つからない
- Kaggle認証を確認（[SECURITY.md](SECURITY.md)の推奨方法で設定）
- [docs/REAL_DATA_SETUP.md](docs/REAL_DATA_SETUP.md) の配置パスを確認
- 実データがない場合はデモデータで自動フォールバックします

### 学習が収束しない
- nnU-Netの `progress.png` を確認（通常100-150エポックで収束開始）
- 旧実装では学習率を1/10に下げて再試行

## 📚 詳細ドキュメント

- [ファイル構造説明](docs/FILE_STRUCTURE.md)
- [データ準備ガイド](docs/REAL_DATA_SETUP.md)
- [Runpodsへのファイル転送](docs/upload_to_runpods.md)
- [セキュリティポリシー](SECURITY.md)

## 🤝 貢献

プルリクエスト歓迎です。大きな変更の場合は、まずIssueを作成して変更内容を議論してください。
**認証情報や大容量データを含むコミットは受け付けられません。**

## 📄 ライセンス

このプロジェクトは[Vesuvius Challenge](https://scrollprize.org/)の公式ルールに準拠しています。

## 🙏 謝辞

- Vesuvius Challengeの主催者とコミュニティ
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)、PyTorch、MONAI開発チーム
- Kaggleプラットフォーム

---

**📮 質問・サポート**: GitHubのIssuesページをご利用ください。
