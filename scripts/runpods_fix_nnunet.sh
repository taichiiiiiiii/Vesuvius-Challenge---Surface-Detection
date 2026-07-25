#!/bin/bash
# Runpods環境でnnU-Netエラーを修正するスクリプト

echo "🏛️ Runpods nnU-Net エラー修正スクリプト"
echo "=============================================="
echo ""

# 1. OpenBLASスレッド制限
echo "📌 Step 1: OpenBLASスレッド数を制限..."
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
echo "✅ スレッド数を4に制限しました"
echo ""

# 2. Dataset100のdataset.json修正（SimpleTiffIO削除）
echo "📌 Step 2: dataset.jsonを修正..."
if [ -f "/workspace/nnUNet_data/nnUNet_raw/Dataset100_VesuviusSurface/dataset.json" ]; then
    python3 -c "
import json
json_path = '/workspace/nnUNet_data/nnUNet_raw/Dataset100_VesuviusSurface/dataset.json'
with open(json_path, 'r') as f:
    config = json.load(f)

# SimpleTiffIO関連を削除
if 'ioclass' in config:
    del config['ioclass']
    print('  削除: ioclass設定')

if 'imageio' in config:
    del config['imageio']
    print('  削除: imageio設定')

# file_endingを確認
config['file_ending'] = '.nii.gz'

with open(json_path, 'w') as f:
    json.dump(config, f, indent=2)
print('✅ dataset.json修正完了')
"
else
    echo "⚠️ dataset.jsonが見つかりません"
fi
echo ""

# 3. TIFF→NIfTI変換（必要な場合）
echo "📌 Step 3: TIFF→NIfTI変換チェック..."
python3 -c "
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

dataset_path = Path('/workspace/nnUNet_data/nnUNet_raw/Dataset100_VesuviusSurface')
if not dataset_path.exists():
    print('❌ Dataset100が見つかりません')
    sys.exit(1)

images_dir = dataset_path / 'imagesTr'
tiff_files = list(images_dir.glob('*.tif'))
nifti_files = list(images_dir.glob('*.nii.gz'))

print(f'  TIFFファイル: {len(tiff_files)}個')
print(f'  NIfTIファイル: {len(nifti_files)}個')

if len(tiff_files) > 0 and len(nifti_files) == 0:
    print('⚠️ TIFF→NIfTI変換が必要です')
    print('  以下のPythonスクリプトを実行してください:')
    print('')
    print('from tifffile import imread')
    print('import nibabel as nib')
    print('import numpy as np')
    print('# ... 変換コード ...')
elif len(nifti_files) > 0:
    print('✅ NIfTIファイルが既に存在します')
else:
    print('⚠️ 画像ファイルが見つかりません')
"
echo ""

# 4. 前処理コマンド生成
echo "📌 Step 4: 修正済み前処理コマンド"
echo "=============================================="
echo ""
echo "以下のコマンドを実行してください:"
echo ""
echo "# 前処理（3d_lowres設定、T4 GPU向け）"
echo "nnUNetv2_plan_and_preprocess -d 100 -c 3d_lowres -np 4"
echo ""
echo "# または、前処理をスキップして学習"
echo "nnUNetv2_train 100 3d_lowres all -tr nnUNetTrainer_250epochs"
echo ""
echo "=============================================="
echo "✅ 修正スクリプト完了！"