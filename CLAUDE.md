# Vesuvius Challenge - 3D Surface Detection Project

## プロジェクト概要

古代ローマのヴェスヴィオ火山噴火で埋もれた巻物の3D CTスキャンデータから、パピルス表面のインク痕跡を検出するディープラーニングモデルの開発プロジェクトです。

### 現在の実装: nnU-Net v2
- **フレームワーク**: nnU-Net v2 - 医療画像セグメンテーションの業界標準
- **アーキテクチャ**: ResNetエンコーダー付きU-Net (nnUNetPlannerResEncM)
- **学習戦略**: ゼロからの学習、250エポック（Kaggleベストプラクティス準拠）
- **旧実装**: PyTorch 3D CNN / SwinUNETR（`notebooks/training/`, `notebooks/runpods/` に参考用として保持）

## ファイル構成

```
.
├── README.md                                    # プロジェクト概要
├── SECURITY.md                                  # セキュリティポリシー
├── requirements.txt                             # 依存パッケージ
├── notebooks/
│   ├── nnunet/
│   │   └── vesuvius_nnunet_runpods.ipynb        # ⭐ メイン学習ノートブック（nnU-Net v2）
│   ├── training/                                # 旧実装（PyTorch 3D CNN / SwinUNETR）
│   ├── inference/                               # 推論・提出ファイル生成
│   └── runpods/                                 # 旧実装（Runpodsオールインワン版）
├── src/
│   ├── unified_data_loader.py                   # 統合データローダー
│   └── download_kaggle_data.py                  # Kaggleデータ自動取得
├── scripts/
│   ├── runpods_safe_setup.sh                    # Runpods環境構築
│   ├── runpods_fix_nnunet.sh                    # nnU-Netエラー一括修正
│   ├── convert_tiff_to_nifti.py                 # TIFF→NIfTI変換
│   ├── fix_nnunet_cv_error.py                   # Cross-validationエラー修正
│   └── fix_nnunet_io_error.py                   # SimpleTiffIOエラー修正
└── docs/                                        # 詳細ドキュメント
```

## セキュリティ上の注意（コード変更時に必ず守ること）

- **認証情報（kaggle.json、APIキー）をコミットしない** — `.gitignore` で除外済みだが、コードやノートブックのセルに直接書かないこと
- 認証は環境変数 `KAGGLE_USERNAME` / `KAGGLE_KEY`、または `~/.kaggle/kaggle.json`・`/workspace/kaggle.json`（Runpods）を使用
- `kaggle.json` をリポジトリ配下（カレントディレクトリ）に書き込むコードを追加しない
- ノートブックはコミット前に出力セルをクリアする
- 詳細は `SECURITY.md` を参照

## 技術スタック

### コアライブラリ
- **nnU-Net v2**: 医療画像セグメンテーションフレームワーク
- **PyTorch**: ディープラーニングバックエンド
- **nibabel**: NIfTI/医療画像フォーマット処理
- **tifffile**: TIFF画像の読み書き

### モデルアーキテクチャ (nnU-Net v2)
- **エンコーダー**: ResNetブロック (nnUNetPlannerResEncM)
- **デコーダー**: U-Net構造（スキップ接続付き）
- **入力**: 3D CTボリューム（パッチサイズはGPU依存）
  - A6000: (96, 96, 96)
  - T4: (64, 64, 64)
- **出力クラス**: 3クラス
  - 0: 背景
  - 1: パピルス表面
  - 2: 無視領域（ラベルなし）

### 損失関数
- **Dice Loss + Cross Entropy Loss** の組み合わせ
- Ignore Label (クラス2) のサポート
- 部分的にラベル付けされたデータでの学習に対応

## データ処理

### データソース
1. **前処理済みデータ（推奨）**
   - `jirkaborovec/vesuvius-surface-nnunet-preprocessed` (91.8GB)
   - 前処理済みで学習時間を大幅短縮（1-2時間→0分）

2. **生データ**
   - `vesuvius-challenge-surface-detection` (25GB)
   - train_images/*.tif (3D CTボリューム)
   - train_labels/*.tif (セグメンテーションマスク)

### データ拡張（nnU-Net自動設定）
- 空間的変換: 回転、反転、弾性変形
- 強度変換: ガンマ補正、ノイズ追加、明度シフト
- nnU-Netが自動的に最適な拡張を選択

## 学習設定

### GPU別最適化設定
| GPU | VRAM | Configuration | パッチサイズ | エポック数 | 推定時間 |
|-----|------|---------------|-------------|-----------|----------|
| A6000 | 48GB | 3d_fullres | (96,96,96) | 250 | 12-15時間 |
| RTX 4090 | 24GB | 3d_fullres | (80,80,80) | 250 | 15-20時間 |
| T4 | 16GB | 3d_lowres | (64,64,64) | 250 | 20-25時間 |

### 学習パラメータ（nnU-Net自動設定）
```python
# nnU-Netが自動的に以下を設定:
- 学習率スケジュール: PolyLR (initial_lr * (1 - epoch/max_epochs)^0.9)
- バッチサイズ: GPU メモリに基づいて自動調整
- データ拡張: データセット統計に基づいて最適化
- ネットワーク深さ: データの解像度に応じて調整
```

### 推奨設定
- **Planner**: nnUNetPlannerResEncM (ResNetエンコーダー、中サイズ)
- **Configuration**: 3d_fullres (高GPU) / 3d_lowres (低GPU)
- **Fold**: all (全データで学習、クロスバリデーションなし)
- **エポック数**: 250 (Kaggle実績値)

## Runpodsでの実行

### セットアップ手順
1. **Network Volume作成** (50GB以上推奨)
2. **Podにアタッチ** (`/workspace/persistent_storage`)
3. **Kaggle認証設定**:
```bash
# kaggle.jsonを/workspace/に配置（リポジトリ内には置かない）
chmod 600 /workspace/kaggle.json
```
4. **ノートブック実行**:
```bash
# Jupyter起動
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

# notebooks/nnunet/vesuvius_nnunet_runpods.ipynb を開いて実行
```

### 自動機能
- **データダウンロード**: Kaggle APIで自動取得
- **GPU最適化**: GPU種類を自動検出して設定調整
- **前処理済みデータ優先**: 91.8GBデータで時間短縮
- **永続化**: Network Volumeに結果を自動保存

## 推論とメトリクス

### 推論コマンド
```bash
nnUNetv2_predict -d 100 -c 3d_fullres -f all \
    -i /path/to/test_images \
    -o /path/to/predictions \
    -p nnUNetResEncUNetMPlans \
    -tr nnUNetTrainer_250epochs
```

### 評価メトリクス
- **Dice Score**: セグメンテーション精度の主要指標
- **Ignore Label対応**: クラス2（ラベルなし領域）を除外して評価
- **Cross-validation**: fold="all"で全データ学習、検証なし

## トラブルシューティング

### メモリ不足エラー (OOM)
- **解決策**: `config="3d_lowres"`に変更
- パッチサイズを小さく: (96,96,96) → (64,64,64)
- バッチサイズは常に1（nnU-Net推奨）

### データ見つからないエラー
- **Kaggle認証確認**: 環境変数またはkaggle.jsonの配置と権限（chmod 600）
- **前処理済みデータ優先**: 91.8GBのデータセットを自動取得

### nnU-Net固有のエラー
- **SimpleTiffIO / I/Oエラー**: `python scripts/fix_nnunet_io_error.py`
- **Cross-validation (n_splits) エラー**: `python scripts/fix_nnunet_cv_error.py`
- **一括修正**: `bash scripts/runpods_fix_nnunet.sh`

### 学習が収束しない
- **progress.png確認**: 通常100-150エポックで収束開始
- **ResNetエンコーダー**: 通常U-Netより安定した学習曲線

## パフォーマンスチューニング

### GPUメモリ最適化
```python
torch.cuda.set_per_process_memory_fraction(0.95)
torch.backends.cudnn.benchmark = True
```

### OpenBLASスレッド制限（前処理時の安定化）
```bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## 実験管理

### チェックポイント（nnU-Net標準）
```
nnUNet_results/Dataset100_VesuviusSurface/
└── nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__3d_fullres/
    └── fold_all/
        ├── checkpoint_best.pth
        ├── checkpoint_final.pth
        └── progress.png
```

## コスト最適化

### Runpods料金目安
| GPU | 時間単価 | 250エポック推定時間 | 推定コスト |
|-----|----------|-------------------|-----------|
| RTX 3090 | $0.5/h | 15-20h | $8-10 |
| RTX 4090 | $0.7/h | 15-20h | $11-14 |
| A6000 | $0.8/h | 12-15h | $10-12 |

### Network Volume
- 50GB: 約$5/月
- データ永続化で再学習時のコスト削減

## 今後の改善案

1. **モデルアーキテクチャ**
   - UNETRやSwinUNETRの検証
   - アンサンブル学習の実装

2. **データ処理**
   - より高度なデータ拡張
   - ハードネガティブマイニング

3. **学習効率化**
   - 混合精度学習の完全実装
   - グラディエント蓄積による大バッチ学習

4. **推論最適化**
   - TensorRTやONNXへの変換
   - バッチ推論の実装

## 参考リンク

- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
- [Vesuvius Challenge](https://scrollprize.org/)
- [Runpods Documentation](https://docs.runpods.io/)

## 注意事項

- このプロジェクトは研究・教育目的です
- 大規模学習には適切なGPUリソースが必要です
- Network Volumeなしでの実行はデータ損失のリスクがあります
- 自動停止機能を必ず有効にして課金を防いでください
- 認証情報の取り扱いは `SECURITY.md` に従ってください
