# Vesuvius Challenge - Surface Detection

ヴェスヴィオ火山で炭化した古代パピルス巻物の3D CTスキャンから、文字が書かれた表面を検出する3D医用画像セグメンテーションプロジェクト。

## 概要

このリポジトリは、[Vesuvius Challenge](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection)のための3Dセグメンテーションモデルの実装を含んでいます。炭化したパピルスの3D CTスキャンデータから、インクが存在する薄い層を正確に検出することが目的です。

## 特徴

- **3D医用画像セグメンテーション**: [Medic-AI](https://github.com/innat/medic-ai)ライブラリを使用
- **マルチバックエンド対応**: JAX、PyTorch、TensorFlowで動作可能
- **高度な損失関数**: Dice-CE Loss + Centerline-Dice Lossの組み合わせ
- **Sliding Window Inference**: 大きな3Dボリュームの正確な推論

## ファイル構成

### メインノートブック

| ファイル | 説明 | 推奨環境 |
|---------|------|----------|
| `train-vesuvius-surface-3d-detection-on-tpu.ipynb` | オリジナルのTPU学習コード | Kaggle TPU |
| `train_original_minimal.ipynb` | 元のコードから最小限の変更版 | GPU (24GB+) |
| `train_minimal_fixed.ipynb` | GPU/CPU対応の実用版 | GPU (8-16GB) |

### サポートファイル

| ファイル | 説明 |
|---------|------|
| `train_gpu_minimal.py` | スタンドアロンPythonスクリプト |
| `train_pure_pytorch.py` | Pure PyTorch実装 |
| `train_minimal.ipynb` | シンプルなPyTorch実装 |

## 環境要件

### 必須パッケージ

```bash
pip install keras>=3.0
pip install medicai
pip install tensorflow  # tf.data用
pip install torch  # PyTorchバックエンド使用時
```

### 推奨GPU環境

| GPU | VRAM | 推奨input_shape | 推奨ノートブック |
|-----|------|-----------------|-----------------|
| RTX 3060 | 12GB | (64, 64, 64) | train_minimal_fixed.ipynb |
| RTX 3070 | 8GB | (32, 64, 64) | train_minimal_fixed.ipynb |
| RTX 3080 | 10GB | (64, 64, 64) | train_minimal_fixed.ipynb |
| RTX 3090 | 24GB | (128, 128, 128) | train_original_minimal.ipynb |
| T4 (Colab) | 16GB | (64, 96, 96) | train_minimal_fixed.ipynb |

## 使用方法

### 1. 環境設定

```python
# バックエンドの選択
import os
os.environ["KERAS_BACKEND"] = "torch"  # または "jax", "tensorflow"
```

### 2. データの準備

TFRecord形式のデータが必要です。データパスを環境に合わせて修正してください：

```python
# Kaggle環境
all_tfrec = glob.glob("/kaggle/input/vesuvius-tfrecords/*.tfrec")

# ローカル環境
all_tfrec = glob.glob("./data/*.tfrec")
```

### 3. 学習の実行

```python
# GPU/メモリに応じて調整
input_shape = (64, 64, 64)  # VRAMに合わせて調整
batch_size = 2  # 通常1-2
epochs = 100
```

## モデルアーキテクチャ

### 利用可能なモデル

1. **SegFormer** (デフォルト)
   - 軽量で高速
   - Transformerベース
   - パラメータ数: 約3.8M

2. **TransUNet**
   - CNN + Transformerハイブリッド
   - 医用画像で高実績

3. **UNETR++**
   - 最新の3D Transformerアーキテクチャ
   - 高精度だが計算コスト大

## 主な特徴

### 損失関数

```python
# 複合損失関数
combined_loss = SparseDiceCELoss + SparseCenterlineDiceLoss
```

- **Dice-CE Loss**: 一般的なセグメンテーション精度
- **Centerline-Dice Loss**: 薄い構造物の検出精度向上

### データ拡張

- 幾何学的変換: ランダムクロップ、フリップ、回転
- 強度変換: 正規化、強度シフト
- 空間的変換: CutOut（ランダムマスキング）

### Sliding Window Inference

大きな3Dボリュームを小さなパッチに分割して推論し、ガウシアン重み付けで結合：

```python
swi = SlidingWindowInference(
    model,
    roi_size=input_shape,
    overlap=0.5,
    mode='gaussian'
)
```

## トラブルシューティング

### メモリ不足エラー

```python
# 入力サイズを削減
input_shape = (32, 64, 64)  # より小さく

# バッチサイズを削減
batch_size = 1

# Centerline-Dice反復数を削減
cldice_loss = SparseCenterlineDiceLoss(iters=10)  # 50→10
```

### 学習が遅い

- CenterlineDiceのitersを削減（50→10-20）
- 入力サイズを削減
- mixed precisionを有効化

## 結果

このモデルは、炭化したパピルスの3D CTスキャンから文字が書かれた薄い表面を検出します。Diceスコアで評価され、Sliding Window Inferenceにより全体ボリュームでの正確な予測が可能です。

## 参考文献

- [Vesuvius Challenge](https://scrollprize.org/)
- [Medic-AI Documentation](https://github.com/innat/medic-ai)
- [Centerline Dice Loss](https://github.com/jocpae/clDice)

## ライセンス

MIT License

## Author

- GitHub: [@taichiiiiiiii](https://github.com/taichiiiiiiii)

---

*This project is part of the Vesuvius Challenge - an effort to read the Herculaneum Papyri using machine learning.*