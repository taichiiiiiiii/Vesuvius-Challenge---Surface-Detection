# Vesuvius Challenge - ファイル構造

## 📁 リポジトリ構成

```
.
├── README.md                    # プロジェクト概要・使用方法
├── CLAUDE.md                    # Claude Code 用プロジェクトコンテキスト
├── SECURITY.md                  # セキュリティポリシー（認証情報の扱い方）
├── requirements.txt             # Python依存パッケージ
│
├── notebooks/
│   ├── nnunet/                  # ⭐ 現行実装（nnU-Net v2）
│   │   └── vesuvius_nnunet_runpods.ipynb   # メイン学習パイプライン
│   │
│   ├── training/                # 旧実装（PyTorch / SwinUNETR）
│   │   ├── main_training.ipynb          # 3D CNN学習（UNet3D / ResNet3D）
│   │   ├── swinunetr_training.ipynb     # SwinUNETR学習
│   │   └── swinunetr_v2.ipynb           # SwinUNETR改良版
│   │
│   ├── inference/               # 推論・提出
│   │   └── inference.ipynb              # 提出ファイル生成
│   │
│   └── runpods/                 # Runpods向けオールインワン版（旧実装）
│       ├── runpods_complete.ipynb
│       ├── runpods_standalone.ipynb
│       ├── runpods_training.ipynb
│       ├── runpods_swinunetr_v2_complete.ipynb
│       └── swinunetr_runpods_complete.ipynb
│
├── src/                         # 共通Pythonモジュール
│   ├── __init__.py
│   ├── unified_data_loader.py   # 統合データローダー（実データ/デモ自動切替）
│   └── download_kaggle_data.py  # Kaggleデータ自動ダウンロード
│
├── scripts/                     # セットアップ・修正スクリプト
│   ├── runpods_safe_setup.sh        # Runpods環境構築
│   ├── runpods_fix_nnunet.sh        # nnU-Netエラー一括修正
│   ├── convert_tiff_to_nifti.py     # TIFF→NIfTI変換
│   ├── fix_nnunet_cv_error.py       # Cross-validationエラー修正
│   └── fix_nnunet_io_error.py       # SimpleTiffIOエラー修正
│
└── docs/                        # ドキュメント
    ├── FILE_STRUCTURE.md        # このファイル
    ├── REAL_DATA_SETUP.md       # 実データ準備ガイド
    └── upload_to_runpods.md     # Runpodsへのファイル転送方法
```

## 🎯 どのノートブックを使うべきか

| 目的 | 使用ファイル |
|------|-------------|
| **本命の学習（推奨）** | `notebooks/nnunet/vesuvius_nnunet_runpods.ipynb` |
| ローカルで軽く試す | `notebooks/training/main_training.ipynb` |
| SwinUNETRを試す | `notebooks/training/swinunetr_v2.ipynb` |
| 推論・提出ファイル生成 | `notebooks/inference/inference.ipynb` |
| Runpodsで1ファイル完結（旧版） | `notebooks/runpods/runpods_complete.ipynb` |

## 🔒 リポジトリに含めてはいけないもの

以下は `.gitignore` で除外されています。**手動でも追加しないでください**（詳細は `SECURITY.md`）:

- `kaggle.json` / `.kaggle/`（Kaggle APIキー）
- `.env` 系ファイル
- `data/`（データセット本体）、`*.tif` 等の大容量データ
- `models/`、`*.pth` 等の学習済み重み
