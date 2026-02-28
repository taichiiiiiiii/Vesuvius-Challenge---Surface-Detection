"""
統合データローダー - 実データとデモデータの両方に対応
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import os

class VesuviusDataset(Dataset):
    """Vesuvius Challenge用の統合データセット"""
    
    def __init__(self, 
                 split='train',
                 volume_size=(96, 96, 64),
                 num_samples=30,
                 data_path: Optional[str] = None):
        """
        Args:
            split: 'train' or 'val'
            volume_size: (H, W, D) ボリュームサイズ
            num_samples: 生成するサンプル数
            data_path: 実データのパス（オプション）
        """
        self.split = split
        self.volume_size = volume_size
        self.num_samples = num_samples
        self.data_path = data_path
        
        # データ生成
        self.volumes, self.labels = self._create_data()
        
    def _create_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """データ生成（実データまたはデモデータ）"""
        
        # 実データ確認
        if self._check_real_data():
            print(f"✅ 実データ使用: {self.data_path}")
            return self._load_real_data()
        else:
            print(f"🎭 デモデータ生成中 ({self.split})")
            return self._generate_demo_data()
    
    def _check_real_data(self) -> bool:
        """実データの存在確認（Runpods環境対応）"""
        if not self.data_path:
            # MCPでダウンロード試行
            try:
                print("📥 Kaggleデータダウンロード試行中...")
                # この関数が既に呼ばれている場合はスキップ
                if not hasattr(self, '_download_attempted'):
                    self._download_attempted = True
                    import subprocess
                    result = subprocess.run(
                        ["python", "-c", 
                         "from mcp__kaggle__prepare_kaggle_dataset import prepare_kaggle_dataset; "
                         "prepare_kaggle_dataset('vesuvius-challenge-surface-detection')"],
                        capture_output=True, text=True, timeout=10
                    )
            except:
                pass
            
            # デフォルトパスを確認
            default_paths = [
                "/workspace/vesuvius-challenge-surface-detection",
                "/content/vesuvius-challenge-surface-detection",
                "./data/vesuvius-challenge-surface-detection",
                "./vesuvius-challenge-surface-detection",
                "../input/vesuvius-challenge-surface-detection"
            ]
            
            for default_path in default_paths:
                path = Path(default_path)
                if path.exists():
                    train_images = path / "train_images"
                    if train_images.exists():
                        tiff_files = list(train_images.glob("*.tif*"))
                        if len(tiff_files) > 0:
                            self.data_path = default_path
                            print(f"✅ 実データ自動検出: {default_path}")
                            return True
            
            # データダウンロードを実行
            print("📥 データダウンロードを手動で実行してください:")
            print("   kaggle competitions download -c vesuvius-challenge-surface-detection -p ./data")
            return False
        
        path = Path(self.data_path)
        if path.exists():
            # train_imagesディレクトリをチェック
            train_images = path / "train_images"
            if train_images.exists():
                tiff_files = list(train_images.glob("*.tif*"))
                return len(tiff_files) > 0
        return False
    
    def _load_real_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """実データのロード"""
        try:
            from PIL import Image
            import cv2
        except ImportError:
            print("⚠️ PIL/cv2が利用不可 - デモデータにフォールバック")
            return self._generate_demo_data()
        
        volumes = []
        labels = []
        
        data_path = Path(self.data_path)
        train_images_dir = data_path / "train_images"
        train_labels_dir = data_path / "train_labels"
        
        # TIFFファイルリスト取得
        tiff_files = sorted(list(train_images_dir.glob("*.tif")))
        
        if len(tiff_files) == 0:
            print("⚠️ TIFFファイルなし - デモデータ使用")
            return self._generate_demo_data()
        
        print(f"📊 {len(tiff_files)}個のTIFFファイル発見")
        
        H, W, D = self.volume_size
        num_volumes = min(self.num_samples, max(1, len(tiff_files) // D))
        
        for vol_idx in range(num_volumes):
            # スライス選択
            start_idx = (vol_idx * D) % max(1, len(tiff_files) - D)
            selected_files = tiff_files[start_idx:start_idx + D]
            
            if len(selected_files) < D:
                # 不足分は循環
                selected_files = selected_files + tiff_files[:D-len(selected_files)]
            
            volume_slices = []
            label_slices = []
            
            for tiff_file in selected_files:
                try:
                    # 画像読み込み
                    img = np.array(Image.open(tiff_file), dtype=np.float32)
                    
                    # グレースケール変換
                    if len(img.shape) == 3:
                        img = img.mean(axis=2)
                    
                    # リサイズ
                    img = cv2.resize(img, (W, H))
                    
                    # 正規化
                    img = (img - img.mean()) / (img.std() + 1e-8)
                    
                    # ラベル処理
                    if train_labels_dir.exists():
                        label_file = train_labels_dir / tiff_file.name
                        if label_file.exists():
                            label = np.array(Image.open(label_file), dtype=np.uint8)
                            if len(label.shape) == 3:
                                label = label.mean(axis=2)
                            label = cv2.resize(label, (W, H), interpolation=cv2.INTER_NEAREST)
                            label = (label > 127).astype(np.int64)
                        else:
                            # 簡易セグメンテーション
                            label = (img > np.percentile(img, 75)).astype(np.int64)
                    else:
                        # 簡易セグメンテーション
                        label = (img > np.percentile(img, 75)).astype(np.int64)
                    
                    volume_slices.append(img)
                    label_slices.append(label)
                    
                except Exception as e:
                    print(f"⚠️ {tiff_file.name}読み込みエラー: {e}")
                    # ダミースライス
                    volume_slices.append(np.random.randn(H, W).astype(np.float32))
                    label_slices.append(np.zeros((H, W), dtype=np.int64))
            
            # 3Dボリューム構築
            volume = np.stack(volume_slices, axis=2)
            label = np.stack(label_slices, axis=2)
            
            volumes.append(volume)
            labels.append(label)
            
            print(f"  ✅ ボリューム{vol_idx+1}: {volume.shape}, 前景{(label==1).mean():.2%}")
        
        print(f"✅ {len(volumes)}個の実データボリューム作成完了")
        return volumes, labels
    
    def _generate_demo_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """高品質デモデータ生成"""
        volumes = []
        labels = []
        
        H, W, D = self.volume_size
        
        for i in range(self.num_samples):
            # リアルなボリューム生成
            volume = np.random.randn(H, W, D).astype(np.float32)
            
            # 複雑なテクスチャ追加
            for z in range(D):
                # ノイズとパターン
                x, y = np.meshgrid(np.linspace(0, 3*np.pi, H), 
                                  np.linspace(0, 3*np.pi, W))
                pattern = np.sin(x + i) * np.cos(y + z/10)
                volume[:, :, z] += pattern * 0.5
                
                # ランダムノイズ
                volume[:, :, z] += np.random.randn(H, W) * 0.2
            
            # 正規化 (-1, 1)
            volume = (volume - volume.mean()) / (volume.std() + 1e-8)
            volume = np.clip(volume, -3, 3)
            
            # セグメンテーションラベル生成
            label = np.zeros((H, W, D), dtype=np.int64)
            
            # 複数の前景領域を作成
            num_regions = np.random.randint(3, 8)
            for _ in range(num_regions):
                # ランダムな中心点
                cx = np.random.randint(H//4, 3*H//4)
                cy = np.random.randint(W//4, 3*W//4)
                cz = np.random.randint(D//4, 3*D//4)
                
                # ランダムなサイズ
                size = np.random.randint(5, 15)
                
                # 3D楕円体を作成
                for x in range(max(0, cx-size), min(H, cx+size)):
                    for y in range(max(0, cy-size), min(W, cy+size)):
                        for z in range(max(0, cz-size//2), min(D, cz+size//2)):
                            dist = ((x-cx)**2 + (y-cy)**2 + (z-cz)**2*4) / size**2
                            if dist < 1:
                                label[x, y, z] = 1
            
            volumes.append(volume)
            labels.append(label)
        
        return volumes, labels
    
    def __len__(self):
        return len(self.volumes)
    
    def __getitem__(self, idx):
        volume = torch.FloatTensor(self.volumes[idx])
        label = torch.LongTensor(self.labels[idx])
        
        # (H, W, D) -> (C=1, H, W, D)
        volume = volume.unsqueeze(0)
        
        return {
            'data': volume,
            'target': label
        }


def create_data_loaders(volume_size=(96, 96, 64),
                       batch_size=4,
                       train_samples=24,
                       val_samples=6,
                       data_path: Optional[str] = None,
                       num_workers=0):
    """
    データローダー作成（Runpods環境対応）
    
    Args:
        volume_size: ボリュームサイズ (H, W, D)
        batch_size: バッチサイズ
        train_samples: 訓練サンプル数
        val_samples: 検証サンプル数
        data_path: 実データパス（オプション - 自動検出）
        num_workers: ワーカー数
        
    Returns:
        train_loader, val_loader
    """
    
    print("🚀 データローダー作成開始（Runpods環境）...")
    
    # データセット作成
    train_dataset = VesuviusDataset(
        split='train',
        volume_size=volume_size,
        num_samples=train_samples,
        data_path=data_path
    )
    
    val_dataset = VesuviusDataset(
        split='val',
        volume_size=volume_size,
        num_samples=val_samples,
        data_path=data_path
    )
    
    # データローダー作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"📊 データセットサイズ:")
    print(f"   訓練: {len(train_dataset)}サンプル")
    print(f"   検証: {len(val_dataset)}サンプル")
    print(f"✅ データローダー作成完了!")
    print(f"   訓練バッチ数: {len(train_loader)}")
    print(f"   検証バッチ数: {len(val_loader)}")
    
    # テスト
    sample_batch = next(iter(train_loader))
    print(f"\n🧪 データローダーテスト:")
    print(f"   データ形状: {sample_batch['data'].shape}")
    print(f"   ラベル形状: {sample_batch['target'].shape}")
    print(f"   データ範囲: [{sample_batch['data'].min():.3f}, {sample_batch['data'].max():.3f}]")
    
    fg_ratio = (sample_batch['target'] == 1).float().mean()
    print(f"   前景比率: {fg_ratio:.3f}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # テスト実行
    print("🧪 統合データローダーテスト")
    
    train_loader, val_loader = create_data_loaders(
        volume_size=(64, 64, 32),
        batch_size=2,
        train_samples=10,
        val_samples=2
    )
    
    print("\n✅ テスト完了!")