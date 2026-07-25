# 📤 Runpodsへのファイル転送方法

Runpods 上でこのプロジェクトを動かすために必要なファイルの転送方法をまとめます。

## 転送が必要なもの

| ファイル | 転送先 | 備考 |
|----------|--------|------|
| このリポジトリ | `/workspace/` | `git clone` が最も簡単 |
| `kaggle.json` | `/workspace/kaggle.json` | ⚠️ リポジトリのクローン先ディレクトリ内には置かない |

> ⚠️ **セキュリティ**: `kaggle.json` の扱いは [SECURITY.md](../SECURITY.md) を必ず確認してください。
> 転送後は `chmod 600 /workspace/kaggle.json` を実行し、Podの共有・スナップショット公開時は削除してください。

## 方法1: git clone（リポジトリ本体・推奨）

```bash
cd /workspace
git clone https://github.com/taichiiiiiiii/Vesuvius-Challenge---Surface-Detection.git
cd Vesuvius-Challenge---Surface-Detection
bash scripts/runpods_safe_setup.sh
```

## 方法2: Jupyter Lab経由でアップロード（kaggle.json等の小さいファイル・推奨）

1. Jupyter Lab 左サイドバーのファイルブラウザで `/workspace` に移動
2. アップロードボタン（↑）をクリック
3. ローカルの `kaggle.json` を選択
4. ターミナルで権限を設定:

```bash
chmod 600 /workspace/kaggle.json
```

## 方法3: SCP（ローカルマシンから）

Runpods の Pod 詳細画面で SSH 接続情報（IP・ポート）を確認してから:

```bash
scp -P <SSH_PORT> kaggle.json root@<RUNPODS_IP>:/workspace/
```

## 方法4: runpodctl（Runpods公式CLI）

```bash
# ローカル側
runpodctl send kaggle.json

# Pod側（表示されたコードを使用）
runpodctl receive <one-time-code>
```

## 転送後の確認

```bash
ls -la /workspace/kaggle.json          # 存在と権限(-rw-------)を確認
cd /workspace/Vesuvius-Challenge---Surface-Detection
ls notebooks/nnunet/                   # メインノートブックの存在確認
```

## 学習の開始

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
# → notebooks/nnunet/vesuvius_nnunet_runpods.ipynb を開いて実行
```

> 💡 Jupyter Lab を外部公開する場合はトークン認証を無効化しないでください。

## トラブルシューティング

### 権限エラー

```bash
chmod 600 /workspace/kaggle.json
```

### `kaggle: command not found`

```bash
pip install kaggle
```

### ディスク容量不足

- Network Volume (50GB以上) を Pod にアタッチしてください
- 生データは約25GB、前処理済みデータは約92GBです
