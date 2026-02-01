# Runpods Setup Guide for Vesuvius Challenge PyTorch Training

RunpodsでVesuvius Challenge PyTorchノートブックを実行するための完全セットアップガイド

## 🚀 Quick Start

### 1. Runpodsでポッドを作成

1. **[Runpods](https://www.runpods.io/)にログイン**
2. **"Deploy"をクリック**
3. **推奨GPU設定:**

| GPU | VRAM | 推奨設定 | 時間単価目安 |
|-----|------|----------|-------------|
| RTX 3080 | 10GB | input_shape=(64,64,64) | $0.3-0.4/h |
| RTX 3090 | 24GB | input_shape=(128,128,128) | $0.4-0.6/h |
| RTX 4090 | 24GB | input_shape=(128,128,128) | $0.6-0.8/h |
| A100 80GB | 80GB | input_shape=(128,128,128), batch_size=4 | $1.5-2.5/h |

4. **Templateを選択:**
   - **PyTorch 2.0** または **RunPod PyTorch**
   - **Jupyter Lab** が含まれているものを選択

### 2. 環境の準備

ポッドが起動したら、ターミナルで以下を実行:

```bash
# 1. 必要パッケージのインストール
pip install keras>=3.0
pip install git+https://github.com/innat/medic-ai.git
pip install tensorflow  # tf.dataのみ使用
pip install matplotlib seaborn tqdm

# 2. PyTorchが正しくインストールされていることを確認
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 3. ノートブックとデータの準備

```bash
# GitHubからノートブックを取得
git clone https://github.com/taichiiiiiiii/Vesuvius-Challenge---Surface-Detection.git
cd "Vesuvius-Challenge---Surface-Detection"

# Jupyter Labを起動 (すでに起動していれば不要)
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

## 📁 データセットの準備

### Option 1: Kaggle Dataset (推奨)

```bash
# 1. Kaggle APIをインストール
pip install kaggle

# 2. Kaggle認証設定
# Kaggle -> Account -> API -> Create New API Token
# kaggle.jsonファイルをダウンロード
mkdir ~/.kaggle
# kaggle.jsonをアップロード (Runpods File Manager使用)
chmod 600 ~/.kaggle/kaggle.json

# 3. Vesuvius TFRecordデータセットをダウンロード
kaggle datasets download -d your-username/vesuvius-tfrecords
unzip vesuvius-tfrecords.zip -d ./data/
```

### Option 2: Direct Upload

```bash
# RunpodsのFile Managerを使用してTFRecordファイルを直接アップロード
mkdir -p ./data/
# TFRecordファイル (*.tfrec) を ./data/ にアップロード
```

### Option 3: Google Drive Mount

```bash
# Google Driveマウント (データが大きい場合)
pip install gdown
# Google DriveのファイルIDを使用
gdown --id YOUR_DRIVE_FILE_ID
```

## ⚙️ ノートブック設定の調整

### GPU メモリに応じた設定調整

ノートブックのCell 10を以下のように修正:

```python
# === GPU メモリ別推奨設定 ===

# RTX 3080 (10GB) の場合
input_shape = (64, 64, 64)
batch_size = 1 * total_device
epochs = 200

# RTX 3090/4090 (24GB) の場合
input_shape = (96, 96, 96)  # または (128, 128, 128)
batch_size = 1 * total_device
epochs = 200

# A100 (80GB) の場合
input_shape = (128, 128, 128)
batch_size = 2 * total_device  # より大きなバッチサイズ
epochs = 200
```

### データパスの修正

ノートブックのCell 17を以下のように修正:

```python
# Runpods環境用のデータパス
all_tfrec = sorted(
    glob.glob("./data/*.tfrec"),  # Kaggleパスから変更
    key=lambda x: int(x.split("_")[-1].replace(".tfrec", ""))
)

# データが見つからない場合のフォールバック
if not all_tfrec:
    print("TFRecord files not found in ./data/")
    print("Please upload TFRecord files to ./data/ directory")
    print("Available files:", glob.glob("./data/*"))
```

### 高速化設定 (オプション)

```python
# Cell 34の損失関数を軽量化 (必要に応じて)
cldice_loss_fn = SparseCenterlineDiceLoss(
    from_logits=False, 
    num_classes=num_classes,
    target_class_ids=1,
    ignore_class_ids=2,
    iters=25  # 50から25に削減 (高速化)
)

# エポック数を調整 (テスト用)
epochs = 50  # 200から50に削減
```

## 🔧 Runpods固有の最適化

### 1. 永続ストレージの設定

```bash
# 永続ストレージをマウント (有料プランの場合)
# Network Storage を作成し、ポッド作成時にアタッチ
# モデルと結果を永続ストレージに保存
ln -s /workspace/persistent_storage ./models
```

### 2. 自動保存の強化

ノートブックに以下のセルを追加:

```python
# 定期保存設定 (Runpods用)
import shutil
from pathlib import Path

def setup_runpods_saving():
    """Runpods用の自動保存設定"""
    
    # 保存ディレクトリの作成
    save_dirs = ['./checkpoints', './results', './logs']
    for dir_path in save_dirs:
        Path(dir_path).mkdir(exist_ok=True)
    
    # 定期チェックポイント保存 (10エポック毎)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath='./checkpoints/model_epoch_{epoch:03d}_dice_{val_dice:.4f}.h5',
        monitor='val_dice',
        mode='max',
        save_best_only=False,
        save_freq=10 * steps_per_epoch,  # 10エポック毎
        verbose=1
    )
    
    return checkpoint_callback

# コールバックリストに追加
enhanced_callbacks.append(setup_runpods_saving())
```

### 3. メモリ監視の強化

```python
# Runpods用メモリ監視
class RunpodsMonitorCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved
            
            elapsed = time.time() - self.start_time
            epoch_time = elapsed / (epoch + 1)
            remaining = epoch_time * (epochs - epoch - 1)
            
            print(f"Epoch {epoch+1} - GPU: {allocated:.2f}GB used, {free:.2f}GB free")
            print(f"Time: {elapsed/3600:.1f}h elapsed, {remaining/3600:.1f}h remaining")
            
            # メモリ不足警告
            if allocated > torch.cuda.get_device_properties(0).total_memory / 1024**3 * 0.9:
                print("⚠️  GPU memory usage is high! Consider reducing input_shape or batch_size.")

# コールバックに追加
enhanced_callbacks.append(RunpodsMonitorCallback())
```

## 🚨 トラブルシューティング

### CUDA/PyTorch関連

```bash
# CUDAバージョン確認
nvidia-smi

# PyTorchとCUDAの互換性確認
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"

# CUDA再インストール (必要に応じて)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### メモリエラー

```python
# Cell 10の設定を小さく
input_shape = (32, 64, 64)  # より小さく
batch_size = 1
cldice_loss_fn = SparseCenterlineDiceLoss(iters=10)  # より少なく
```

### データ読み込みエラー

```bash
# TFRecordファイルの確認
ls -la ./data/
python -c "
import glob
files = glob.glob('./data/*.tfrec')
print(f'Found {len(files)} TFRecord files')
for f in files[:5]: print(f)
"
```

### 接続が切れる場合

```bash
# tmux/screenを使用 (長時間訓練用)
tmux new-session -d -s training
tmux send-keys -t training 'cd /workspace && jupyter lab' Enter

# セッション復帰
tmux attach -t training
```

## 💰 コスト最適化のヒント

### 1. 適切なGPU選択

```python
# コスト効率の良い設定
# RTX 3080: $0.3/h, input_shape=(64,64,64)
# 24時間で約$7.2, Diceスコア 0.7-0.8 期待

# 高性能設定  
# A100: $2/h, input_shape=(128,128,128), batch_size=4
# 12時間で約$24, Diceスコア 0.8+ 期待
```

### 2. 段階的学習

```python
# Phase 1: 小さなサイズで高速プロトタイプ (2-3時間)
input_shape = (64, 64, 64)
epochs = 50

# Phase 2: 本格訓練 (8-12時間)
input_shape = (128, 128, 128)  
epochs = 200
```

### 3. Auto-Stop設定

```python
# 早期停止を積極的に使用
keras.callbacks.EarlyStopping(
    monitor='val_dice',
    patience=15,  # より短く
    mode='max',
    restore_best_weights=True,
    min_delta=0.001  # 改善閾値
)
```

## 🎯 実行手順まとめ

1. **Runpodsでポッド作成** (RTX 3090推奨)
2. **環境セットアップ** (pip install)
3. **リポジトリクローン** 
4. **データアップロード** (TFRecordファイル)
5. **設定調整** (GPU メモリに応じて)
6. **ノートブック実行** (train-vesuvius-surface-3d-detection-pytorch-backend.ipynb)
7. **結果保存** (checkpoints, models)

## ⏱️ 期待される実行時間

| GPU | 設定 | 時間目安 | コスト目安 |
|-----|------|----------|------------|
| RTX 3080 | (64,64,64), 200ep | 18-24h | $5-10 |
| RTX 3090 | (96,96,96), 200ep | 12-18h | $5-11 |
| RTX 4090 | (128,128,128), 200ep | 8-12h | $5-10 |
| A100 | (128,128,128), 200ep, bs=4 | 6-8h | $10-20 |

これでRunpodsで効率的に学習を実行できます！
