# Vesuvius Challenge - 3D Surface Detection

**nnU-Net v2による最先端医療画像セグメンテーション** を用いた古代パピルス表面検出プロジェクト

[![Runpods](https://img.shields.io/badge/Runpods-Ready-blue)](https://runpods.io)
[![nnU-Net](https://img.shields.io/badge/nnU--Net-v2-green)](https://github.com/MIC-DKFZ/nnUNet)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org)

## 🎯 プロジェクト概要

古代ローマのヴェスヴィオ火山噴火で埋もれた巻物の3D CTスキャンから、パピルス表面のインク痕跡を検出する深層学習モデルです。

### 🏆 **現在の実装: nnU-Net v2 (2024年12月)**
- **フレームワーク**: nnU-Net v2 - 医療画像セグメンテーションの業界標準
- **アーキテクチャ**: ResNetエンコーダー付きU-Net (nnUNetPlannerResEncM)
- **学習戦略**: ゼロからの学習、250エポック
- **データ処理**: 自動前処理パイプライン、NIfTI形式対応

## 📁 プロジェクト構成

```
.
├── vesuvius_nnunet_runpods.ipynb           # 🎯 メイン学習ノートブック
├── fix_nnunet_cv_error.py                  # Cross-validation エラー修正
├── convert_tiff_to_nifti.py                # TIFF→NIfTI変換ツール
├── fix_nnunet_io_error.py                  # SimpleTiffIO エラー修正
├── runpods_fix_nnunet.sh                   # Runpods環境修正スクリプト
├── requirements.txt                        # 依存パッケージ
├── CLAUDE.md                              # プロジェクト詳細仕様書
└── RUNPODS_DEPLOYMENT_GUIDE.md            # Runpods詳細ガイド
```

## 🚀 クイックスタート

### ステップ1: Runpodsセットアップ
1. **[Runpods](https://runpods.io)** でアカウント作成
2. **Network Volume作成** (100GB以上推奨)
3. **GPU Pod起動** (A6000/RTX 4090推奨)

### ステップ2: 環境準備
```bash
# 1. リポジトリクローン
git clone https://github.com/taichiiiiiiii/Vesuvius-Challenge---Surface-Detection.git
cd "Vesuvius-Challenge---Surface-Detection"

# 2. Kaggle認証設定
chmod 600 /workspace/kaggle.json  # kaggle.jsonをアップロード後

# 3. Jupyter起動
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
```

### ステップ3: 学習実行
**メインノートブック `vesuvius_nnunet_runpods.ipynb` を開いて Run All**

## 📊 技術スタック

### モデルアーキテクチャ
| コンポーネント | 詳細 |
|-------------|------|
| **フレームワーク** | nnU-Net v2 |
| **エンコーダー** | ResNetブロック (nnUNetPlannerResEncM) |
| **デコーダー** | U-Net構造（スキップ接続付き） |
| **入力サイズ** | 96×96×96 (A6000), 64×64×64 (T4) |
| **出力クラス** | 3クラス (背景/パピルス表面/無視領域) |

### 損失関数
- **Dice Loss + Cross Entropy Loss** の組み合わせ
- Ignore Label (クラス2) のサポート
- 部分的ラベルデータでの学習に対応

### GPU別推奨設定
| GPU | VRAM | Configuration | パッチサイズ | 推定時間 |
|-----|------|---------------|-------------|----------|
| A6000 | 48GB | 3d_fullres | (96,96,96) | 12-15時間 |
| RTX 4090 | 24GB | 3d_fullres | (80,80,80) | 15-20時間 |
| T4 | 16GB | 3d_lowres | (64,64,64) | 20-25時間 |

## 💡 主要機能

### ✅ **自動GPU検出・最適化**
- VRAM容量に応じた自動設定調整
- A6000/RTX 4090/T4 対応済み

### ✅ **軽量プログレス表示**
- エポック全体 + エポック内バッチ進捗
- 軽量テキストベース（HTML不使用）
- リアルタイムDice/Loss/ETA表示

### ✅ **エラー自動修正**
- SimpleTiffIO エラー → NIfTI形式に自動変換
- OpenBLAS スレッドエラー → 自動制限
- Cross-validation エラー → 自動修正

### ✅ **Checkpoint管理**
- 自動保存: best/final/latest
- 進捗監視: progress.png, training_log.txt
- ディスク容量管理機能

## 📈 データセット

### データソース（自動ダウンロード）
1. **前処理済みデータ（推奨）**: `jirkaborovec/vesuvius-surface-nnunet-preprocessed` (91.8GB)
2. **生データ**: `vesuvius-challenge-surface-detection` (25GB)

### データ拡張（nnU-Net自動設定）
- 空間的変換: 回転、反転、弾性変形
- 強度変換: ガンマ補正、ノイズ追加
- nnU-Netが自動的に最適な拡張を選択

## 💰 コスト見積もり

### Network Volume
| サイズ | 月額料金 | 推奨度 |
|--------|----------|--------|
| 100GB | $10 | ⚠️ 最小限 |
| 150GB | $15 | ✅ 推奨 |
| 200GB | $20 | ✅ 理想 |

### GPU学習コスト
| GPU | 時間単価 | 学習時間 | 推定コスト |
|-----|----------|----------|------------|
| RTX 3080 | $0.3/h | 18-24h | $5-7 |
| RTX 4090 | $0.5/h | 12-18h | $6-9 |
| A6000 | $1.5/h | 8-12h | $12-18 |

## 🔧 トラブルシューティング

### メモリ不足エラー
```python
CONFIGURATION = "3d_lowres"  # よりメモリ効率的
```

### データ見つからないエラー
```bash
# Kaggle認証確認
ls ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 学習が収束しない
- **progress.png確認**: 通常100-150エポックで収束開始
- **ResNetエンコーダー**: 通常U-Netより安定した学習曲線

## 📚 詳細ドキュメント

- **[CLAUDE.md](CLAUDE.md)** - 技術仕様・アーキテクチャ詳細
- **[RUNPODS_DEPLOYMENT_GUIDE.md](RUNPODS_DEPLOYMENT_GUIDE.md)** - Runpods詳細セットアップ

## 🎉 期待される結果

- **Dice Score**: 0.7-0.8+ (A6000設定)
- **学習曲線**: 安定した収束パターン
- **推論時間**: ~1-2分/ケース
- **モデルサイズ**: ~90MB (checkpoint_best.pth)

## 🔗 参考リンク

- [nnU-Net Repository](https://github.com/MIC-DKFZ/nnUNet)
- [Vesuvius Challenge](https://scrollprize.org/)
- [Runpods Documentation](https://docs.runpods.io/)

## ⚠️ 重要な注意事項

- Network Volumeなしでの実行はデータ損失のリスクがあります
- 自動停止機能を必ず有効にして課金を防いでください
- このプロジェクトは研究・教育目的です