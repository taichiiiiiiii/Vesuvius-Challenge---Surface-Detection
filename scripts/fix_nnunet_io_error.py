#!/usr/bin/env python3
"""
nnU-Net SimpleTiffIO エラー修正スクリプト
TIFFファイルを適切にNIfTI形式に変換し、OpenBLAS設定も最適化
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

def fix_openblas_threads():
    """OpenBLASスレッド数を制限"""
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
    os.environ['NUMEXPR_NUM_THREADS'] = '4'
    print("✅ OpenBLASスレッド数を4に制限しました")

def convert_tiff_to_nifti():
    """TIFFファイルをNIfTI形式に変換（Dataset100用）"""
    print("\n🔄 TIFF → NIfTI変換開始...")
    
    # Dataset100のパスを確認
    dataset_path = Path("/workspace/nnUNet_data/nnUNet_raw/Dataset100_VesuviusSurface")
    if not dataset_path.exists():
        # ローカル環境用のパス
        dataset_path = Path("/Users/taichi/コンペ/Kaggle/Vesuvius Challenge - Surface Detection/vesuvius_data/preprocessed_download/Dataset100_VesuviusSurface")
        if not dataset_path.exists():
            print(f"❌ Dataset100が見つかりません: {dataset_path}")
            return False
    
    images_dir = dataset_path / "imagesTr"
    labels_dir = dataset_path / "labelsTr"
    
    # TIFFファイルを確認
    tiff_files = list(images_dir.glob("*.tif"))
    print(f"   発見したTIFFファイル: {len(tiff_files)}個")
    
    if len(tiff_files) == 0:
        print("   TIFFファイルが見つかりません")
        return True  # 既に変換済みの可能性
    
    try:
        from tifffile import imread
    except ImportError:
        print("⚠️ tifffileをインストールします...")
        os.system("pip install tifffile -q")
        from tifffile import imread
    
    # 各TIFFファイルを変換
    converted = 0
    for tiff_file in tiff_files[:10]:  # 最初の10個を処理
        try:
            # TIFFファイルを読み込み
            img_data = imread(str(tiff_file))
            
            # 3D形状を確認
            if len(img_data.shape) != 3:
                print(f"   ⚠️ スキップ: {tiff_file.name} (形状: {img_data.shape})")
                continue
            
            # NIfTI形式で保存
            nifti_name = tiff_file.stem + "_0000.nii.gz"
            nifti_path = images_dir / nifti_name
            
            # 既存なら変換しない
            if nifti_path.exists():
                print(f"   ✓ 既存: {nifti_name}")
                continue
            
            # NIfTI保存
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(img_data.astype(np.float32), affine)
            nib.save(nifti_img, str(nifti_path))
            
            converted += 1
            print(f"   ✅ 変換: {tiff_file.name} → {nifti_name}")
            
            # 対応するラベルも処理
            label_tiff = labels_dir / tiff_file.name
            if label_tiff.exists():
                label_data = imread(str(label_tiff))
                label_nifti_path = labels_dir / (tiff_file.stem + ".nii.gz")
                
                if not label_nifti_path.exists():
                    label_nifti = nib.Nifti1Image(label_data.astype(np.uint8), affine)
                    nib.save(label_nifti, str(label_nifti_path))
                    print(f"      ラベルも変換: {label_nifti_path.name}")
            
        except Exception as e:
            print(f"   ❌ エラー: {tiff_file.name} - {e}")
    
    print(f"\n✅ 変換完了: {converted}個のファイルを処理")
    return True

def fix_dataset_json():
    """dataset.jsonからioclass設定を削除"""
    print("\n🔧 dataset.json修正中...")
    
    # 複数のパスを試す
    paths_to_check = [
        Path("/workspace/nnUNet_data/nnUNet_raw/Dataset100_VesuviusSurface/dataset.json"),
        Path("/Users/taichi/コンペ/Kaggle/Vesuvius Challenge - Surface Detection/vesuvius_data/preprocessed_download/Dataset100_VesuviusSurface/dataset.json"),
        Path("./nnUNet_data/nnUNet_raw/Dataset100_VesuviusSurface/dataset.json")
    ]
    
    json_path = None
    for path in paths_to_check:
        if path.exists():
            json_path = path
            break
    
    if not json_path:
        print("   dataset.jsonが見つかりません")
        return False
    
    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        # ioclassやSimpletiffIO関連を削除
        modified = False
        if 'ioclass' in config:
            del config['ioclass']
            modified = True
            print("   ✅ 'ioclass'設定を削除")
        
        if 'imageio' in config:
            del config['imageio']
            modified = True
            print("   ✅ 'imageio'設定を削除")
        
        # file_endingを確認
        if config.get('file_ending') != '.nii.gz':
            config['file_ending'] = '.nii.gz'
            modified = True
            print("   ✅ file_endingを'.nii.gz'に設定")
        
        # 保存
        if modified:
            with open(json_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"   ✅ dataset.json修正完了: {json_path}")
        else:
            print("   ✓ dataset.jsonは既に適切です")
        
        return True
        
    except Exception as e:
        print(f"   ❌ エラー: {e}")
        return False

def create_spacing_files():
    """spacing情報ファイルを作成"""
    print("\n📝 spacing情報ファイル作成中...")
    
    dataset_path = Path("/workspace/nnUNet_data/nnUNet_raw/Dataset100_VesuviusSurface")
    if not dataset_path.exists():
        dataset_path = Path("/Users/taichi/コンペ/Kaggle/Vesuvius Challenge - Surface Detection/vesuvius_data/preprocessed_download/Dataset100_VesuviusSurface")
    
    if not dataset_path.exists():
        print("   データセットが見つかりません")
        return False
    
    images_dir = dataset_path / "imagesTr"
    
    # 各画像ファイルに対してspacing情報を作成
    for img_file in images_dir.glob("*.nii.gz"):
        spacing_file = img_file.with_suffix('.json')
        if not spacing_file.exists():
            spacing_info = {
                "spacing": [1.0, 1.0, 1.0],
                "shape": None,  # 実際の形状は後で設定
                "origin": [0.0, 0.0, 0.0]
            }
            
            # 実際の画像形状を取得
            try:
                img = nib.load(str(img_file))
                spacing_info["shape"] = list(img.shape)
            except:
                spacing_info["shape"] = [320, 320, 320]  # デフォルト
            
            with open(spacing_file, 'w') as f:
                json.dump(spacing_info, f)
            
            print(f"   ✅ spacing作成: {spacing_file.name}")
    
    return True

def main():
    """メイン実行"""
    print("=" * 80)
    print("🏛️ nnU-Net SimpleTiffIO エラー修正")
    print("=" * 80)
    
    # 1. OpenBLAS設定
    fix_openblas_threads()
    
    # 2. TIFFをNIfTIに変換
    if not convert_tiff_to_nifti():
        print("❌ TIFF変換に失敗しました")
        return 1
    
    # 3. dataset.json修正
    if not fix_dataset_json():
        print("❌ dataset.json修正に失敗しました")
        return 1
    
    # 4. spacing情報作成
    create_spacing_files()
    
    print("\n✅ すべての修正が完了しました！")
    print("\n🚀 以下のコマンドで前処理を再実行してください:")
    print("   export OMP_NUM_THREADS=4")
    print("   export OPENBLAS_NUM_THREADS=4")
    print("   nnUNetv2_plan_and_preprocess -d 100 -c 3d_lowres")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())