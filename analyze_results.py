# analyze_results.py

import yaml
import pandas as pd

def analyze_and_print_tables(results_file="results/experiment_outputs.yaml"):
    """
    YAML結果ファイルを読み込み、サマリーテーブルを生成して表示する。
    """
    
    # --- 1. YAMLファイルの読み込み ---
    try:
        with open(results_file, 'r') as f:
            docs = list(yaml.safe_load_all(f))
    except FileNotFoundError:
        print(f"エラー: 結果ファイル '{results_file}' が見つかりません。")
        return

    # 最初のドキュメント（config）と、2つ目以降（結果）を分離
    if not docs or len(docs) < 2:
        print("結果ファイルに十分なデータがありません。")
        return
        
    config_snapshot = docs[0]
    results_list = [item for doc in docs[1:] for item in doc] # 全ての結果をフラットなリストに

    if not results_list:
        print("実験結果が見つかりません。")
        return

    # --- 2. データをpandas DataFrameに変換 ---
    # 必要なデータだけを抽出してリストに格納
    data_for_df = []
    for result in results_list:
        exp = result.get('experiment', {})
        data_for_df.append({
            'pair': tuple(exp.get('pair', [])),
            'error_type': exp.get('error_type'),
            'error_rate': exp.get('error_rate'),
            'test_accuracy': exp.get('accuracy', {}).get('test'),
            'f1_score': exp.get('f1_score'),
            'best_epoch': exp.get('best_epoch')
        })
    
    df = pd.DataFrame(data_for_df)
    
    # --- 3. ペアごとにループし、ピボットテーブルを作成して表示 ---
    # DataFrameをペアでグループ化
    grouped = df.groupby('pair')

    print("======================================================")
    print("           実験結果サマリーテーブル")
    print("======================================================")
    print("指標: test_accuracy (テストデータでの正解率)\n")

    for pair, group_df in grouped:
        # pivot_tableを使って、指定した形式の表を自動生成
        summary_table = group_df.pivot_table(
            index='error_rate',      # 表の「行」になる列
            columns='error_type',    # 表の「列」になる列
            values='test_accuracy'   # 表の中身の「値」になる列
        )
        
        # 見やすいように整形
        summary_table.index = [f"{x*100:.1f}%" for x in summary_table.index]
        
        print(f"--- ペア: {pair[0]} vs {pair[1]} ---")
        print(summary_table)
        print("\n" + "-"*50 + "\n")

if __name__ == '__main__':
    # pandasの表示オプションを設定して、全ての列を省略せずに表示
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 100)
    
    analyze_and_print_tables()