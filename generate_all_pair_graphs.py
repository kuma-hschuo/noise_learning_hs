# generate_all_pair_graphs.py

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm # 進捗バーを表示するためにtqdmをインポート

# --- 定数定義 ---
RESULTS_FILE = "results/experiment_outputs.yaml"
# 成果物を保存する専用のサブディレクトリ
OUTPUT_DIR = "summary_report/all_pair_graphs/"

def load_data_to_dataframe(filepath):
    """
    指定されたYAML結果ファイルを読み込み、分析に適したPandas DataFrameを返す。
    """
    print(f"'{filepath}' から実験結果を読み込んでいます...")
    try:
        with open(filepath, 'r') as f:
            docs = list(yaml.safe_load_all(f))
            if len(docs) < 2:
                print("エラー: ファイルに実験結果が含まれていません。")
                return None
            results_list = [item for doc in docs[1:] for item in doc]
    except FileNotFoundError:
        print(f"エラー: 結果ファイル '{filepath}' が見つかりません。")
        return None

    data = [{'pair': tuple(r['experiment']['pair']), 'error_type': r['experiment']['error_type'],
             'error_rate': r['experiment']['error_rate'], 'test_accuracy': r['experiment']['accuracy']['test']}
            for r in results_list]
    return pd.DataFrame(data)

def create_performance_plot(df, title, filename, output_dir):
    """
    DataFrameから性能曲線グラフを生成し、指定されたディレクトリに保存する。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_df = df.pivot_table(index='error_rate', columns='error_type', values='test_accuracy')
    
    for error_type in plot_df.columns:
        ax.plot(plot_df.index, plot_df[error_type], marker='o', label=error_type)
        
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Error Rate", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_ylim(0.45, 1.02)
    ax.legend(title="Error Type")
    ax.set_xticklabels([f'{x:.0%}' for x in ax.get_xticks()])
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def main():
    """
    メイン処理: 全てのペアの性能曲線グラフを生成する。
    """
    df = load_data_to_dataframe(RESULTS_FILE)
    if df is None:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"全ペアのグラフを '{OUTPUT_DIR}' フォルダに保存します...")

    # DataFrameをペアでグループ化し、各ペアに対して処理を行う
    # tqdmでラップして進捗バーを表示する
    grouped = df.groupby('pair')
    for pair, group_df in tqdm(grouped, total=len(grouped), desc="Generating Pair Graphs"):
        
        # グラフのタイトルとファイル名を生成
        pair_str = f"{pair[0]} vs {pair[1]}"
        title = f"Performance Curve for Pair: {pair_str}"
        filename = f"curve_pair_{pair[0]}_{pair[1]}.png"
        
        # グラフ作成関数を呼び出し
        create_performance_plot(group_df, title, filename, OUTPUT_DIR)

    print("\n--- 全てのペアのグラフ生成が完了しました ---")

if __name__ == '__main__':
    main()