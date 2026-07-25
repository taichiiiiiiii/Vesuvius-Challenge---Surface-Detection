# セキュリティポリシー / Security Policy

## 認証情報の取り扱い（最重要）

このプロジェクトは Kaggle API を使用します。**認証情報は絶対にリポジトリにコミットしないでください。**

### やってはいけないこと

- `kaggle.json` をリポジトリ内（プロジェクトディレクトリ配下）に置いたままコミットする
- ノートブックのセルにユーザー名・APIキーを直接書いたまま保存・コミットする
- APIキーを含むノートブックの出力セルを残したまま共有する

### 推奨される認証方法（優先順）

1. **環境変数を使用する**

   ```bash
   export KAGGLE_USERNAME="your_username"
   export KAGGLE_KEY="your_api_key"
   ```

2. **標準の場所に `kaggle.json` を配置する**

   ```bash
   mkdir -p ~/.kaggle
   mv kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Runpods 等のクラウド環境では `/workspace/kaggle.json` を使用する**
   （リポジトリのクローン先ディレクトリには置かない）

   ```bash
   chmod 600 /workspace/kaggle.json
   ```

`.gitignore` には `kaggle.json` / `.kaggle/` / `.env` などの除外設定が入っていますが、
これは最後の防衛線です。最初からリポジトリ外に置くことを徹底してください。

## 誤ってキーをコミット・公開してしまった場合

1. **直ちにキーを無効化する**: [Kaggle Settings](https://www.kaggle.com/settings/account) → API → "Expire API Token"
2. 新しいトークンを発行する
3. リポジトリ履歴からの削除も行う（`git filter-repo` 等）。ただし、一度公開された
   キーは漏洩済みとみなし、**無効化を必ず先に**行うこと

## ノートブック共有時の注意

- コミット前に出力セルをクリアする: `jupyter nbconvert --clear-output --inplace <notebook>.ipynb`
- 出力にアクセストークン・IPアドレス・個人パスが含まれていないか確認する

## クラウド環境（Runpods 等）での注意

- Jupyter Lab を外部公開する際は必ずトークン/パスワードを設定する
  （`--NotebookApp.token=''` のような無効化はしない）
- 使用後は Pod を停止し、不要になった Network Volume 上の `kaggle.json` は削除する
- 外部からスクリプトをダウンロードして実行する場合は、URL と内容を確認してから実行する

## 脆弱性の報告

このリポジトリのコードにセキュリティ上の問題を見つけた場合は、
公開 Issue ではなく GitHub の [Private vulnerability reporting](https://docs.github.com/ja/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) か、
リポジトリオーナーへの直接連絡で報告してください。

## 依存パッケージ

- `requirements.txt` のパッケージは定期的に更新してください
- 既知の脆弱性チェック: `pip install pip-audit && pip-audit -r requirements.txt`
