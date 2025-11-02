# create_report_tables.py

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

def create_and_save_tables(results_file="results/experiment_outputs.yaml", output_dir="tables/"):
    """
    YAML結果ファイルを読み込み、サマリーテーブルを生成してコンソールに表示し、
    さらに画像ファイルとして指定されたディレクトリに保存する。
    """
    
    # --- 1. YAMLファイルの読み込み ---
    try:
        with open(results_file, 'r') as f:
            docs = list(yaml.safe_load_all(f))
    except FileNotFoundError:
        print(f"エラー: 結果ファイル '{results_file}' が見つかりません。")
        return
    
    if len(docs) < 2:
        print("結果ファイルに十分なデータがありません。")
        return
        
    results_list = [item for doc in docs[1:] for item in doc]
    if not results_list:
        print("実験結果が見つかりません。")
        return

    # --- 2. データをpandas DataFrameに変換 ---
    data_for_df = [{'pair': tuple(r['experiment']['pair']), 'error_type': r['experiment']['error_type'],
                    'error_rate': r['experiment']['error_rate'], 'test_accuracy': r['experiment']['accuracy']['test']}
                   for r in results_list]
    df = pd.DataFrame(data_for_df)
    
    # --- 3. 画像保存用ディレクトリの作成 ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"テーブル画像は '{output_dir}' フォルダに保存されます。\n")

    # --- 4. ペアごとにループし、テーブルを作成して表示＆保存 ---
    grouped = df.groupby('pair')

    print("======================================================")
    print("           実験結果サマリーテーブル")
    print("======================================================")
    print("指標: test_accuracy (テストデータでの正解率)\n")

    for pair, group_df in grouped:
        summary_table = group_df.pivot_table(index='error_rate', columns='error_type', values='test_accuracy')
        summary_table.index = [f"{x*100:.1f}%" for x in summary_table.index]
        
        # --- コンソールへの表示 ---
        print(f"--- ペア: {pair[0]} vs {pair[1]} ---")
        print(summary_table)
        print("\n" + "-"*50 + "\n")

        # --- 画像として保存するロジック ---
        
        fig, ax = plt.subplots(figsize=(8, 7))

        ax.axis('off')

        table = ax.table(
            cellText=summary_table.values,
            colLabels=summary_table.columns,
            rowLabels=summary_table.index,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)

        ax.set_title(f"Pair: {pair[0]} vs {pair[1]} | Test Accuracy", fontsize=16, pad=40)

        filename = f"table_pair_{pair[0]}_{pair[1]}.png"
        save_path = os.path.join(output_dir, filename)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"全{len(grouped)}個のテーブルが画像として保存されました。")

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 100)
    create_and_save_tables()