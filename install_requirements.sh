#!/bin/bash

# このスクリプトは、仮想環境がアクティベートされた状態で実行してください。
#じゃないと本環境の依存関係が壊れます
#
# 使い方:
# 1. 仮想環境をアクティベートする (source venv/bin/activate など)
# 2. このスクリプトに実行権限を与える (chmod +x install_requirements.sh)
# 3. スクリプトを実行する (./install_requirements.sh)

set -e

echo "--- Python依存ライブラリのインストールを開始します ---"

# requirements.txtからライブラリをインストール
pip install -r requirements.txt

echo ""
echo "--- インストールが正常に完了しました ---"