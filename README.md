# Vesuvius Challenge - Surface Detection (PyTorch Backend)

ヴェスヴィオ火山で炭化した古代パピルス巻物の3D CTスキャンから、文字が書かれた表面を検出する3D医用画像セグメンテーションプロジェクト。

## 概要

このリポジトリは、[Vesuvius Challenge](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection)のための3Dセグメンテーションモデルの**PyTorchバックエンド実装**を含んでいます。炭化したパピルスの3D CTスキャンデータから、インクが存在する薄い層を正確に検出することが目的です。

## 🆕 最新の特徴

- **PyTorch最適化**: 完全にPyTorchバックエンドに最適化された実装
- **GPU効率化**: メモリ管理、CUDA最適化、混合精度学習対応
- **拡張監視**: GPU使用量、訓練進捗、性能ベンチマーク機能
- **エラーハンドリング**: 自動検証、緊急保存、診断機能
- **3D医用画像セグメンテーション**: [Medic-AI](https://github.com/innat/medic-ai)ライブラリを使用
- **高度な損失関数**: Dice-CE Loss + Centerline-Dice Lossの組み合わせ
- **Sliding Window Inference**: ガウシアン重み付けによる大きな3Dボリュームの正確な推論

## 📁 ファイル構成

### 🎯 メインノートブック

| ファイル | 説明 | バックエンド | 推奨環境 |
|---------|------|-------------|----------|
| **`train-vesuvius-surface-3d-detection-pytorch-backend.ipynb`** ⭐️ | **PyTorch最適化版メイン実装** | **PyTorch** | **GPU (8GB+)** |

### 📚 参考ファイル（ローカルのみ）

以下のファイルはローカルには存在しますが、GitHubには含まれていません：

| ファイル | 説明 | 用途 |
|---------|------|------|
| `train-vesuvius-surface-3d-detection-on-tpu.ipynb` | オリジナルTPU版（JAX） | 比較・参考 |
| `train-vesuvius-surface-3d-detection-in-pytorch.ipynb` | PyTorch版（カスタム訓練） | 参考実装 |

## 🚀 環境要件

### 必須パッケージ

```bash
# PyTorchバックエンドで必要
pip install keras>=3.0
pip install medicai
pip install torch torchvision  # PyTorchバックエンド
pip install tensorflow  # tf.data用のみ
```

### 推奨GPU環境

| GPU | VRAM | 推奨input_shape | バッチサイズ | 期待性能 |
|-----|------|-----------------|-------------|----------|
| RTX 3060 | 12GB | (64, 64, 64) | 1-2 | 良好 |
| RTX 3070 | 8GB | (64, 64, 64) | 1 | 良好 |
| RTX 3080 | 10GB | (96, 96, 96) | 1-2 | 優秀 |
| RTX 3090 | 24GB | (128, 128, 128) | 2-4 | 最優 |
| RTX 4080 | 16GB | (96, 96, 96) | 2-3 | 優秀 |
| RTX 4090 | 24GB | (128, 128, 128) | 3-5 | 最優 |
| T4 (Colab) | 16GB | (64, 96, 96) | 1-2 | 良好 |

## 📖 使用方法

### 1. 🔧 環境設定

```python
# PyTorchバックエンドを設定（自動設定されます）
import os
os.environ["KERAS_BACKEND"] = "torch"

# GPU確認
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
```

### 2. 📁 データの準備

TFRecord形式のデータが必要です。データパスを環境に合わせて修正してください：

```python
# Kaggle環境
all_tfrec = glob.glob("/kaggle/input/vesuvius-tfrecords/*.tfrec")

# ローカル環境
all_tfrec = glob.glob("./data/*.tfrec")

# Google Colab
all_tfrec = glob.glob("/content/data/*.tfrec")
```

### 3. 🏃‍♂️ 学習の実行

```python
# GPU/メモリに応じて調整
input_shape = (128, 128, 128)  # VRAMに合わせて調整
batch_size = 1 * total_device  # デバイス数に応じて自動調整
epochs = 200

# 自動的にPyTorch最適化が適用されます
# - GPU メモリ管理（95%使用）
# - CUDA最適化（cudnn.benchmark=True）
# - 混合精度学習サポート
```

## 🏗️ モデルアーキテクチャ

### 利用可能なモデル

1. **SegFormer** (デフォルト) ⭐️
   - 軽量で高速
   - Transformerベース
   - パラメータ数: 約3.8M
   - PyTorch最適化済み

2. **TransUNet**
   - CNN + Transformerハイブリッド
   - 医用画像で高実績

3. **UNETR++**
   - 最新の3D Transformerアーキテクチャ
   - 高精度だが計算コスト大

4. **UPerNet**
   - Pyramid poolingベース
   - マルチスケール特徴抽出

## 🔬 主な特徴

### 損失関数

```python
# 複合損失関数（元の実装を完全保持）
combined_loss = SparseDiceCELoss + SparseCenterlineDiceLoss
```

- **Dice-CE Loss**: 一般的なセグメンテーション精度
- **Centerline-Dice Loss**: 薄い構造物の検出精度向上（50イテレーション）

### データ拡張

元のTPU実装と同等の高度なデータ拡張：

- 幾何学的変換: RandSpatialCrop、RandFlip、RandRotate90、RandRotate
- 強度変換: NormalizeIntensity、RandShiftIntensity
- 空間的変換: RandCutOut（volume mode、num_cuts=5）

### PyTorch特有の最適化

```python
# GPU メモリ最適化
torch.cuda.set_per_process_memory_fraction(0.95)
torch.backends.cudnn.benchmark = True

# 自動監視
PyTorchMonitorCallback()  # GPU使用量の自動追跡
ReduceLROnPlateau()       # 学習率の自動調整
EarlyStopping()          # 早期停止
```

### Sliding Window Inference

ガウシアン重み付けによる高品質な推論：

```python
swi = SlidingWindowInference(
    model,
    roi_size=input_shape,
    overlap=0.5,
    mode='gaussian',  # ガウシアン重み付け
    sw_batch_size=1 * total_device
)
```

## 🔧 トラブルシューティング

### メモリ不足エラー

```python
# 1. 入力サイズを削減
input_shape = (64, 64, 64)  # 128→64

# 2. バッチサイズを削減
batch_size = 1

# 3. Centerline-Dice反復数を削減
cldice_loss = SparseCenterlineDiceLoss(iters=25)  # 50→25
```

### 学習が遅い場合

```python
# 1. より小さなモデルを使用
model = SegFormer(encoder_name='mit_b0')  # 最軽量

# 2. CenterlineDiceのitersを削減
cldice_loss = SparseCenterlineDiceLoss(iters=10)  # 50→10

# 3. 入力サイズを削減
input_shape = (64, 64, 64)

# 4. 混合精度学習が自動的に有効（PyTorch最適化）
```

### データローディングエラー

ノートブックには自動検証機能が含まれています：

```python
# 自動実行される検証
validate_data_loading()           # データローディング検証
validate_model_pytorch_compatibility()  # モデル互換性確認
diagnose_pytorch_issues()         # 一般的な問題の診断
```

## 📊 性能と監視

### 自動ベンチマーク

```python
# 推論速度の自動測定
benchmark_inference()  # 平均時間、GPU使用量を表示

# 訓練履歴の可視化
plot_training_history(history)  # 損失・精度グラフ

# 実験設定の自動保存
save_experiment_config()  # JSON形式で設定保存
```

### GPU使用量監視

各エポック後に自動的にGPU使用量が表示されます：

```
GPU Memory allocated: 7.84 GB
```

## 💾 モデル保存と読み込み

```python
# 複数形式での自動保存
export_pytorch_model()

# 出力ファイル:
# - vesuvius_model_pytorch.keras     # 完全なモデル
# - vesuvius_weights_pytorch.h5      # 重みのみ
# - model_summary_pytorch.txt        # モデル構造
# - experiment_config_pytorch.json   # 実験設定
```

## 🎯 結果

このPyTorchバックエンド実装は：

- **元のTPU実装と同等の精度**: 全ての機能を完全保持
- **GPU効率性の向上**: メモリ使用量最適化、CUDA高速化
- **拡張監視機能**: 訓練進捗、GPU使用量、性能ベンチマーク
- **エラー耐性の向上**: 自動検証、緊急保存、診断機能

Diceスコアで評価され、Sliding Window Inferenceにより全体ボリュームでの正確な予測が可能です。

## 📚 参考文献

- [Vesuvius Challenge](https://scrollprize.org/)
- [Medic-AI Documentation](https://github.com/innat/medic-ai)
- [Centerline Dice Loss](https://github.com/jocpae/clDice)
- [PyTorch Documentation](https://pytorch.org/docs/)

## ⚖️ ライセンス

MIT License

## 👨‍💻 Author

- GitHub: [@taichiiiiiiii](https://github.com/taichiiiiiiii)

---

### 🔄 更新履歴

- **v2.0** (Latest): PyTorchバックエンド最適化版
  - 完全なPyTorch最適化
  - GPU効率化とメモリ管理
  - 拡張監視・診断機能
  - エラーハンドリング強化

- **v1.0**: オリジナルTPU/JAX実装

---

*This project is part of the Vesuvius Challenge - an effort to read the Herculaneum Papyri using machine learning.*