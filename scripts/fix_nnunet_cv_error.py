#!/usr/bin/env python3
"""
nnU-Net v2 Cross-Validation エラー修正統合スクリプト
'n_splits=5 greater than the number of samples: n_samples=3' エラーを完全解決
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import argparse


class NnuNetCVErrorFixer:
    """nnU-Net v2 CV エラー修正クラス"""
    
    def __init__(self, dataset_id: int = 1, verbose: bool = True):
        self.dataset_id = dataset_id
        self.verbose = verbose
        self.dataset_name = f"Dataset{dataset_id:03d}_Vesuvius"
        
        # 環境変数確認・設定
        self.setup_environment()
    
    def setup_environment(self):
        """nnU-Net環境確認・設定"""
        required_vars = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
        
        for var in required_vars:
            if var not in os.environ:
                default_path = f"./nnUNet_data/{var.split('_')[1]}"
                os.environ[var] = default_path
                Path(default_path).mkdir(parents=True, exist_ok=True)
                
                if self.verbose:
                    print(f"🔧 環境変数設定: {var} = {default_path}")
    
    def diagnose_dataset(self) -> dict:
        """データセット診断"""
        if self.verbose:
            print("🔍 データセット診断中...")
        
        raw_path = Path(os.environ['nnUNet_raw'])
        dataset_path = raw_path / self.dataset_name
        
        diagnosis = {
            'dataset_exists': dataset_path.exists(),
            'num_samples': 0,
            'has_dataset_json': False,
            'issues': []
        }
        
        if not dataset_path.exists():
            diagnosis['issues'].append("データセットディレクトリが存在しません")
            return diagnosis
        
        # dataset.json確認
        json_path = dataset_path / "dataset.json"
        if json_path.exists():
            diagnosis['has_dataset_json'] = True
            
            try:
                with open(json_path, 'r') as f:
                    dataset_config = json.load(f)
                    diagnosis['num_samples'] = dataset_config.get('numTraining', 0)
                    
                    if diagnosis['num_samples'] < 5:
                        diagnosis['issues'].append(f"サンプル数不足: {diagnosis['num_samples']} < 5")
                    
            except Exception as e:
                diagnosis['issues'].append(f"dataset.json読み込みエラー: {e}")
        else:
            diagnosis['issues'].append("dataset.jsonが存在しません")
        
        # 実際のファイル数確認
        images_dir = dataset_path / "imagesTr"
        if images_dir.exists():
            actual_files = len(list(images_dir.glob("*.nii.gz")) + list(images_dir.glob("*.npy")))
            if actual_files != diagnosis['num_samples']:
                diagnosis['issues'].append(f"ファイル数不一致: 実際={actual_files}, JSON={diagnosis['num_samples']}")
        
        if self.verbose:
            print(f"   サンプル数: {diagnosis['num_samples']}")
            print(f"   問題数: {len(diagnosis['issues'])}")
        
        return diagnosis
    
    def fix_dataset_json(self, diagnosis: dict) -> bool:
        """dataset.json修正"""
        if not diagnosis['dataset_exists']:
            print("❌ データセットが存在しないため修正不可")
            return False
        
        dataset_path = Path(os.environ['nnUNet_raw']) / self.dataset_name
        json_path = dataset_path / "dataset.json"
        
        if not json_path.exists():
            print("❌ dataset.jsonが存在しないため修正不可")
            return False
        
        if self.verbose:
            print("🔧 dataset.json修正中...")
        
        try:
            with open(json_path, 'r') as f:
                config = json.load(f)
            
            num_samples = diagnosis['num_samples']
            
            # fold数を動的調整
            if num_samples < 5:
                optimal_folds = 1
                config['disable_cross_validation'] = True
            elif num_samples < 10:
                optimal_folds = min(3, num_samples)
                config['disable_cross_validation'] = False
            else:
                optimal_folds = 5
                config['disable_cross_validation'] = False
            
            # カスタム設定追加
            config['vesuvius_custom'] = {
                'optimal_folds': optimal_folds,
                'small_dataset_mode': num_samples < 10,
                'single_fold_mode': num_samples < 5,
                'original_samples': num_samples
            }
            
            # 保存
            with open(json_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            if self.verbose:
                print(f"✅ dataset.json修正完了")
                print(f"   推奨fold数: {optimal_folds}")
                print(f"   CV無効化: {config.get('disable_cross_validation', False)}")
            
            return True
            
        except Exception as e:
            print(f"❌ dataset.json修正エラー: {e}")
            return False
    
    def generate_fixed_training_commands(self, num_samples: int) -> list:
        """修正済み学習コマンド生成"""
        commands = []
        
        # 前処理コマンド
        preprocess_cmd = [
            "nnUNetv2_plan_and_preprocess",
            "-d", str(self.dataset_id),
            "--verify_dataset_integrity"
        ]
        commands.append(("前処理", " ".join(preprocess_cmd)))
        
        # 学習コマンド（サンプル数に応じて調整）
        if num_samples < 5:
            # 単一fold強制
            train_cmd = [
                "nnUNetv2_train",
                str(self.dataset_id),
                "3d_fullres", 
                "0",
                "-tr", "VesuviusSingleFoldTrainer",
                "--disable_cross_validation",
                "--val_freq", "10",
                "--save_freq", "25"
            ]
            commands.append(("学習（単一fold）", " ".join(train_cmd)))
            
        elif num_samples < 10:
            # 3-fold CV
            for fold in [0, 1, 2]:
                train_cmd = [
                    "nnUNetv2_train",
                    str(self.dataset_id),
                    "3d_fullres",
                    str(fold),
                    "-tr", "VesuviusCustomTrainer",
                    "--val_freq", "10"
                ]
                commands.append((f"学習（fold {fold}/3）", " ".join(train_cmd)))
        
        else:
            # 標準5-fold CV
            commands.append(("学習（全fold）", f"nnUNetv2_train {self.dataset_id} 3d_fullres all"))
        
        return commands
    
    def apply_fixes(self) -> bool:
        """全修正を適用"""
        print("🔧 nnU-Net v2 CV エラー修正開始...")
        
        # 1. 診断
        diagnosis = self.diagnose_dataset()
        
        if not diagnosis['dataset_exists']:
            print("❌ データセットが存在しません。まずデータ準備を実行してください。")
            return False
        
        if len(diagnosis['issues']) == 0:
            print("✅ 問題は検出されませんでした")
            return True
        
        print(f"\n🚨 検出された問題:")
        for issue in diagnosis['issues']:
            print(f"   - {issue}")
        
        # 2. dataset.json修正
        if not self.fix_dataset_json(diagnosis):
            print("❌ dataset.json修正に失敗")
            return False
        
        # 3. 修正済みコマンド生成・表示
        commands = self.generate_fixed_training_commands(diagnosis['num_samples'])
        
        print(f"\n🚀 修正済み実行コマンド:")
        for label, cmd in commands:
            print(f"\n{label}:")
            print(f"  {cmd}")
        
        # 4. カスタムトレーナーファイル確認
        current_dir = Path(__file__).parent
        trainer_file = current_dir / "training" / "02_nnunet_v2" / "vesuvius_custom_trainer.py"
        
        if not trainer_file.exists():
            print(f"\n⚠️ カスタムトレーナーファイルが見つかりません: {trainer_file}")
            print("   vesuvius_custom_trainer.pyを同じディレクトリに配置してください")
        else:
            print(f"✅ カスタムトレーナー確認: {trainer_file}")
        
        print(f"\n✅ nnU-Net v2 CV エラー修正完了!")
        print(f"   サンプル数: {diagnosis['num_samples']}")
        print(f"   推奨実行: 上記コマンドを順番に実行してください")
        
        return True


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="nnU-Net v2 Cross-Validation エラー修正")
    parser.add_argument("--dataset_id", type=int, default=1, help="データセットID")
    parser.add_argument("--verbose", action="store_true", help="詳細出力")
    parser.add_argument("--auto_run", action="store_true", help="修正後に自動実行")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🏛️ VESUVIUS CHALLENGE - nnU-Net v2 CV エラー修正ツール")
    print("=" * 80)
    print("'n_splits=5 greater than the number of samples' エラーを解決します\n")
    
    fixer = NnuNetCVErrorFixer(
        dataset_id=args.dataset_id,
        verbose=args.verbose
    )
    
    success = fixer.apply_fixes()
    
    if success and args.auto_run:
        print("\n🚀 自動実行オプションが有効です")
        # TODO: 実際の自動実行ロジックを追加
        print("   手動実行を推奨（現在は自動実行未実装）")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
