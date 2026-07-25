#!/usr/bin/env python3
"""
Convert TIFF files to NIfTI format for nnU-Net
Fixes the "Image shape (320, 320, 320)" error
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import json
from tqdm import tqdm

def convert_dataset_to_nifti(dataset_path: Path):
    """Convert all TIFF files in dataset to NIfTI format"""
    
    print(f"🔄 Converting dataset at: {dataset_path}")
    
    # Process images
    images_dir = dataset_path / "imagesTr"
    if images_dir.exists():
        tiff_files = list(images_dir.glob("*.tif"))
        print(f"Found {len(tiff_files)} TIFF images to convert")
        
        for tiff_file in tqdm(tiff_files, desc="Converting images"):
            # Output NIfTI path
            nifti_file = tiff_file.with_suffix('.nii.gz')
            
            if nifti_file.exists():
                print(f"  Skipping {nifti_file.name} (already exists)")
                continue
            
            try:
                # Load TIFF using tifffile
                import tifffile
                img_data = tifffile.imread(str(tiff_file))
                
                # Ensure 3D shape
                if len(img_data.shape) == 2:
                    img_data = img_data[np.newaxis, :, :]
                
                # Convert to float32
                img_data = img_data.astype(np.float32)
                
                # Create NIfTI with identity affine
                affine = np.eye(4)
                nifti_img = nib.Nifti1Image(img_data, affine)
                
                # Save NIfTI
                nib.save(nifti_img, str(nifti_file))
                
                # Remove original TIFF
                tiff_file.unlink()
                
                print(f"  ✅ Converted: {tiff_file.name} → {nifti_file.name}")
                
            except Exception as e:
                print(f"  ❌ Error converting {tiff_file.name}: {e}")
    
    # Process labels
    labels_dir = dataset_path / "labelsTr"
    if labels_dir.exists():
        tiff_files = list(labels_dir.glob("*.tif"))
        print(f"\nFound {len(tiff_files)} TIFF labels to convert")
        
        for tiff_file in tqdm(tiff_files, desc="Converting labels"):
            # Output NIfTI path
            nifti_file = tiff_file.with_suffix('.nii.gz')
            
            if nifti_file.exists():
                print(f"  Skipping {nifti_file.name} (already exists)")
                continue
            
            try:
                # Load TIFF
                import tifffile
                label_data = tifffile.imread(str(tiff_file))
                
                # Ensure 3D shape
                if len(label_data.shape) == 2:
                    label_data = label_data[np.newaxis, :, :]
                
                # Convert to uint8
                label_data = label_data.astype(np.uint8)
                
                # Create NIfTI
                affine = np.eye(4)
                nifti_img = nib.Nifti1Image(label_data, affine)
                
                # Save NIfTI
                nib.save(nifti_img, str(nifti_file))
                
                # Remove original TIFF
                tiff_file.unlink()
                
                print(f"  ✅ Converted: {tiff_file.name} → {nifti_file.name}")
                
            except Exception as e:
                print(f"  ❌ Error converting {tiff_file.name}: {e}")
    
    # Update dataset.json
    json_path = dataset_path / "dataset.json"
    if json_path.exists():
        print("\n📝 Updating dataset.json...")
        
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        # Update file ending
        config['file_ending'] = '.nii.gz'
        
        # Remove SimpleTiffIO
        if 'overwrite_image_reader_writer' in config:
            del config['overwrite_image_reader_writer']
        if 'ioclass' in config:
            del config['ioclass']
        
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print("  ✅ Updated dataset.json to use NIfTI format")
    
    print("\n✅ Conversion complete!")


def main():
    """Main function for standalone execution"""
    # Try different possible dataset locations
    possible_paths = [
        Path("/workspace/nnUNet_data/nnUNet_raw/Dataset100_VesuviusSurface"),
        Path("./nnUNet_data/nnUNet_raw/Dataset100_VesuviusSurface"),
        Path("/workspace/persistent_storage/nnUNet_data/nnUNet_raw/Dataset100_VesuviusSurface")
    ]
    
    dataset_path = None
    for path in possible_paths:
        if path.exists():
            dataset_path = path
            break
    
    if not dataset_path:
        print("❌ Dataset100 not found in expected locations")
        return 1
    
    # Install tifffile if needed
    try:
        import tifffile
    except ImportError:
        print("Installing tifffile...")
        os.system("pip install tifffile -q")
        import tifffile
    
    # Convert dataset
    convert_dataset_to_nifti(dataset_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())