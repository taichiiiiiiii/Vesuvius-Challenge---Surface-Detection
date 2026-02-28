#!/usr/bin/env python3
"""
Kaggle Vesuvius Challengeデータセット自動ダウンロードスクリプト
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path
import shutil

def check_kaggle_api():
    """Kaggle APIのインストール状態を確認"""
    try:
        import kaggle
        return True
    except ImportError:
        print("⚠️ Kaggle APIが見つかりません。インストールします...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
        return True

def setup_kaggle_credentials():
    """Kaggle認証設定確認（workspace対応）"""
    
    # 複数の場所でkaggle.jsonを探す
    possible_locations = [
        Path.home() / ".kaggle" / "kaggle.json",  # デフォルト
        Path("/workspace") / "kaggle.json",       # Runpods
        Path("/workspace") / ".kaggle" / "kaggle.json",  # Runpods alternative
        Path("./kaggle.json"),                    # カレントディレクトリ
        Path("./.kaggle/kaggle.json"),           # カレントディレクトリ内
        Path("/content") / "kaggle.json",         # Colab
        Path("/kaggle") / "kaggle.json",          # Kaggle Notebooks
    ]
    
    kaggle_json = None
    for location in possible_locations:
        if location.exists():
            kaggle_json = location
            print(f"✅ Kaggle認証ファイル発見: {kaggle_json}")
            break
    
    if not kaggle_json:
        print("⚠️ Kaggle認証ファイルが見つかりません")
        print("\n📝 Kaggle API設定手順:")
        print("1. https://www.kaggle.com/account にアクセス")
        print("2. 'Create New API Token'をクリック")
        print("3. kaggle.jsonをダウンロード")
        print("4. 以下のいずれかの場所に配置:")
        print("   - /workspace/kaggle.json (Runpods)")
        print("   - ~/.kaggle/kaggle.json (ローカル)")
        print("   - ./kaggle.json (カレントディレクトリ)")
        return False
    
    # 権限設定
    try:
        os.chmod(kaggle_json, 0o600)
    except:
        pass  # Windowsなど権限設定できない環境では無視
    
    # 環境変数設定（~/.kaggle以外の場所の場合）
    if kaggle_json.parent.name != ".kaggle" or kaggle_json.parent.parent != Path.home():
        os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_json.parent)
        print(f"📁 KAGGLE_CONFIG_DIR設定: {kaggle_json.parent}")
    
    return True

def download_vesuvius_dataset(output_dir="./data"):
    """Vesuviusデータセットをダウンロード"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset_dir = output_path / "vesuvius-challenge-surface-detection"
    
    # 既にダウンロード済みか確認
    if dataset_dir.exists():
        train_images = dataset_dir / "train_images"
        if train_images.exists():
            tiff_files = list(train_images.glob("*.tif"))
            if len(tiff_files) > 0:
                print(f"✅ データセット既存: {dataset_dir}")
                print(f"   画像数: {len(tiff_files)}")
                return str(dataset_dir)
    
    print("📥 Vesuviusデータセットダウンロード開始...")
    
    try:
        # Kaggle CLIでダウンロード
        cmd = [
            "kaggle", "competitions", "download",
            "-c", "vesuvius-challenge-surface-detection",
            "-p", str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ ダウンロードエラー: {result.stderr}")
            return None
        
        print("✅ ダウンロード完了")
        
        # ZIPファイル解凍
        zip_file = output_path / "vesuvius-challenge-surface-detection.zip"
        
        if zip_file.exists():
            print("📦 解凍中...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            
            # ZIP削除（容量節約）
            zip_file.unlink()
            print("✅ 解凍完了")
        
        # train.zipとtest.zipも解凍
        for subset in ["train", "test"]:
            subset_zip = dataset_dir / f"{subset}.zip"
            if subset_zip.exists():
                print(f"📦 {subset}.zip解凍中...")
                with zipfile.ZipFile(subset_zip, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                subset_zip.unlink()
        
        # データ確認
        train_images = dataset_dir / "train_images"
        if train_images.exists():
            tiff_files = list(train_images.glob("*.tif"))
            print(f"✅ データセット準備完了!")
            print(f"   場所: {dataset_dir}")
            print(f"   画像数: {len(tiff_files)}")
            return str(dataset_dir)
        else:
            print("⚠️ train_imagesディレクトリが見つかりません")
            return None
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def download_with_mcp():
    """MCPサーバー経由でダウンロード（代替手段）"""
    print("🔄 MCPサーバー経由でダウンロード試行...")
    
    # この関数は既にMCPでダウンロード済みの場合に使用
    # mcp__kaggle__prepare_kaggle_dataset が呼ばれた後
    
    # 可能な保存先を確認
    possible_paths = [
        Path.home() / ".kaggle" / "datasets" / "vesuvius-challenge-surface-detection",
        Path("/tmp") / "vesuvius-challenge-surface-detection",
        Path("./data") / "vesuvius-challenge-surface-detection"
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✅ MCPダウンロード済みデータ発見: {path}")
            
            # dataディレクトリにコピー
            target = Path("./data/vesuvius-challenge-surface-detection")
            if not target.exists():
                print(f"📋 データをコピー中: {path} -> {target}")
                shutil.copytree(path, target)
            
            return str(target)
    
    return None

def main():
    """メイン処理"""
    print("=" * 50)
    print("🏺 Vesuvius Challenge データセットダウンローダー")
    print("=" * 50)
    
    # Kaggle API確認
    if not check_kaggle_api():
        print("❌ Kaggle APIのインストールに失敗しました")
        return None
    
    # 認証確認
    if not setup_kaggle_credentials():
        # MCPダウンロードを試行
        mcp_path = download_with_mcp()
        if mcp_path:
            return mcp_path
        
        print("\n⚠️ Kaggle認証なしでは続行できません")
        print("上記の手順に従ってkaggle.jsonを設定してください")
        return None
    
    # ダウンロード実行
    dataset_path = download_vesuvius_dataset()
    
    if not dataset_path:
        # MCPダウンロードを試行
        dataset_path = download_with_mcp()
    
    if dataset_path:
        print("\n" + "=" * 50)
        print("🎉 セットアップ完了!")
        print(f"📂 データセット: {dataset_path}")
        print("=" * 50)
        return dataset_path
    else:
        print("\n❌ データセットのダウンロードに失敗しました")
        return None

if __name__ == "__main__":
    dataset_path = main()
    
    if dataset_path:
        # データ構造を表示
        print("\n📊 データ構造:")
        dataset_dir = Path(dataset_path)
        
        for subdir in ["train_images", "train_labels", "test_images"]:
            path = dataset_dir / subdir
            if path.exists():
                files = list(path.glob("*"))
                print(f"  {subdir}: {len(files)}ファイル")