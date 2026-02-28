# 📤 Runpodsへのファイルアップロード方法

## 現在の状況
✅ 最小限の実装で動作中  
⚠️ 完全な機能を使うには、完全版`improved_transunet.py`のアップロードが必要

---

## 方法1: Jupyter Lab経由でアップロード（推奨）

1. **Jupyter Labのファイルブラウザを使用**
   - 左サイドバーのファイルアイコンをクリック
   - `/workspace`ディレクトリに移動
   - アップロードボタン（↑）をクリック
   - ローカルから以下のファイルを選択：
     - `improved_transunet.py`（必須）
     - `transunet_checkpoint_inference.py`（推論用）

---

## 方法2: ターミナル経由でアップロード

### A. SCPを使用（ローカルマシンから）
```bash
# 単一ファイル
scp improved_transunet.py root@[RUNPODS_IP]:/workspace/

# 複数ファイル
scp improved_transunet.py transunet_checkpoint_inference.py root@[RUNPODS_IP]:/workspace/
```

### B. Runpods内でwget/curlを使用
```bash
# GitHubから直接取得（リポジトリがある場合）
cd /workspace
wget https://raw.githubusercontent.com/[USER]/[REPO]/main/improved_transunet.py

# または、一時的な共有リンクから
curl -o improved_transunet.py [SHARE_LINK]
```

---

## 方法3: コードを直接貼り付け（小規模な場合）

Jupyter Notebookで新しいセルを作成：

```python
# ファイル作成セル
with open('/workspace/improved_transunet.py', 'w') as f:
    f.write('''
# ここに完全なコードを貼り付け
[完全版improved_transunet.pyの内容]
''')
print("✅ Full version uploaded!")
```

---

## 方法4: Google DriveやDropbox経由

```python
# Google Driveからダウンロード
!pip install gdown
!gdown --id [FILE_ID] -O /workspace/improved_transunet.py

# Dropboxから
!wget -O /workspace/improved_transunet.py "[DROPBOX_LINK]?dl=1"
```

---

## 📁 必要なファイル一覧

### 必須ファイル
- `improved_transunet.py` - メインモジュール（約1200行）

### 推奨ファイル
- `transunet_checkpoint_inference.py` - 推論エンジン
- `improved_transunet_training_with_logging.ipynb` - 学習ノートブック
- `transunet_inference_with_checkpoints.ipynb` - 推論ノートブック

---

## ✅ アップロード確認

アップロード後、以下のコードで確認：

```python
import os
from pathlib import Path

# ファイル確認
files_to_check = [
    'improved_transunet.py',
    'transunet_checkpoint_inference.py'
]

print("📁 Checking files in /workspace:")
for file in files_to_check:
    file_path = Path('/workspace') / file
    if file_path.exists():
        size_kb = file_path.stat().st_size / 1024
        print(f"  ✅ {file} ({size_kb:.1f} KB)")
        
        # 行数確認（完全版は1000行以上）
        with open(file_path, 'r') as f:
            lines = len(f.readlines())
        print(f"     Lines: {lines}")
        
        if lines < 100:
            print(f"     ⚠️ This looks like the minimal version")
        else:
            print(f"     ✅ This appears to be the full version")
    else:
        print(f"  ❌ {file} not found")
```

---

## 🚀 アップロード後の手順

1. **モジュール再読み込み**
```python
import importlib
import improved_transunet
importlib.reload(improved_transunet)
print("✅ Module reloaded")
```

2. **機能確認**
```python
from improved_transunet import (
    ImprovedTransUNet,
    ImprovedTransUNetConfig,
    TrainingLogger,
    CheckpointManager,
    MemoryEfficientDataset,  # 完全版のみ
    EfficientTTAPredictor,    # 完全版のみ
    get_optimal_batch_size    # 完全版のみ
)
print("✅ All components imported successfully!")
```

3. **学習開始**
```python
# 設定作成
config = ImprovedTransUNetConfig(
    img_size=256,
    batch_size=16,
    num_epochs=100
)

# モデル初期化
model = ImprovedTransUNet(config)
print("✅ Ready for training with full features!")
```

---

## ⚠️ トラブルシューティング

### ファイルが大きすぎる場合
```bash
# 分割してアップロード
split -b 10M improved_transunet.py improved_transunet_part_
# Runpods内で結合
cat improved_transunet_part_* > improved_transunet.py
```

### 権限エラー
```bash
chmod 644 /workspace/improved_transunet.py
```

### インポートエラーが続く場合
```python
import sys
sys.path.insert(0, '/workspace')
# Pythonを再起動
exec(open('/workspace/improved_transunet.py').read())
```

---

## 📝 まとめ

現在は最小限の実装で動作していますが、完全な機能を利用するには：

1. 完全版`improved_transunet.py`をアップロード
2. モジュールを再読み込み
3. 全機能が利用可能に！

最も簡単な方法は**Jupyter Labのアップロードボタン**を使用することです。