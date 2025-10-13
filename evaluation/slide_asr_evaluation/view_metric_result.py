import os
import sys
import json
import pandas as pd
from loguru import logger as logu

from tqdm import tqdm
from collections import defaultdict
from openpyxl import Workbook
from openpyxl.utils import get_column_letter


def aggregate_metrics(input_path, output_excel):
    """
    Aggregates metrics from a JSONL file with a three-level hierarchy:
    type -> language -> model.
    """
    # Modified data structure to handle the 'type' key
    # Structure: type -> model -> language -> stats
    all_stats = defaultdict(  # Type level (e.g., 'speech', 'dialogue')
        lambda: defaultdict(  # Model level
            lambda: defaultdict(  # Language level
                lambda: {
                    'total_errors': 0,
                    'total_tokens': 0,
                    'ne_total_errors': 0,
                    'ne_total_entities': 0,
                    'total_h': 0,
                    'ne_total_target': 0
                }
            )
        )
    )

    with open(input_path, 'r') as f:
        for line in tqdm(f, desc="Processing lines"):
            data = json.loads(line.strip())
            # Extract type and language, with defaults for safety
            data_type = data.get('type', 'UNKNOWN_TYPE')
            language = data.get('language', 'UNKNOWN_LANG')

            for model, info in data['asr_info'].items():
                wer_info = info['wer_info']
                ne_wer_info = info['ne_wer_info']
                ne_fnr_info = info['ne_fnr_info']

                # Get the specific stats dictionary for the current type, model, and language
                stats = all_stats[data_type][model][language]

                # Update WER stats
                stats['total_errors'] += (
                    wer_info['I'] + wer_info['D'] + wer_info['S']
                )
                stats['total_tokens'] += wer_info['T']
                # Update NE-WER stats
                stats['ne_total_errors'] += (
                    ne_wer_info['I'] + ne_wer_info['D'] + ne_wer_info['S']
                )
                stats['ne_total_entities'] += ne_wer_info['T']
                # Update NE-FNR stats
                stats['total_h'] += ne_fnr_info['H']
                stats['ne_total_target'] += ne_fnr_info['T']

    # --- Calculate Final Metrics ---
    # Structure: type -> model -> language -> {WER, NE-WER, NE-FNR}
    results = defaultdict(lambda: defaultdict(dict))

    for data_type, model_stats in all_stats.items():
        for model, lang_stats in model_stats.items():
            for lang, stats in lang_stats.items():
                wer = (stats['total_errors'] / stats['total_tokens']) if stats['total_tokens'] else 0
                ne_wer = (stats['ne_total_errors'] / stats['ne_total_entities']) if stats['ne_total_entities'] else 0
                ne_fnr = (1 - stats['total_h'] / stats['ne_total_target']) if stats['ne_total_target'] else 0

                results[data_type][model][lang] = {
                    "WER": round(wer, 4),
                    "NE-WER": round(ne_wer, 4),
                    "NE-FNR": round(ne_fnr, 4)
                }

    # --- Generate Excel and Console Output ---
    # Replace with your model names
    # model_list = ["model1", "model1_coarse-grained", "model1_fine-grained", "model2", "model2_coarse-grained", "model2_fine-grained"]
    model_list = json.loads(open(input_path).readlines()[0])['asr_info'].keys()
    columns = ["Model", "WER", "NE-WER", "NE-FNR"]
    
    all_data = []
    for data_type, type_results in sorted(results.items()):
        # Group rows by language for this type
        lang_rows = defaultdict(list)
        all_langs_in_type = set()

        for model in model_list:
            if model in type_results:
                for lang, metrics in type_results[model].items():
                    all_langs_in_type.add(lang)
                    # Handle both "Chinese" and "Mandarin" as the same category for processing
                    display_lang = "Chinese" if lang in ["Chinese", "Mandarin"] else lang
                    
                    lang_rows[display_lang].append({
                        "Model": model,
                        "WER": f'{metrics["WER"] * 100:.2f}',
                        "NE-WER": f'{metrics["NE-WER"] * 100:.2f}',
                        "NE-FNR": f'{metrics["NE-FNR"] * 100:.2f}'
                    })

        # --- Collect data for Excel ---
        for lang, rows in lang_rows.items():
            for row in rows:
                all_data.append({
                    "Type": data_type,
                    "Language": lang,
                    "Model": row["Model"],
                    "WER": row["WER"],
                    "NE-WER": row["NE-WER"],
                    "NE-FNR": row["NE-FNR"]
                })

    # Create a single DataFrame for all data
    df_all = pd.DataFrame(all_data)
    df_all.to_csv(output_excel)
    df_all = df_all.reindex(columns=['Model', 'Language', 'Type', 'WER', 'NE-WER', 'NE-FNR'])
    df_all = df_all.sort_values(by=['Model', 'Language', 'Type'], ascending=[True, False, False])
    logu.info(f"\n{df_all}")
    
    # for i in range(len(df_all)):
    #     row = df_all.iloc[i]
    #     wer, ne_wer, ne_fnr = row['WER'], row['NE-WER'], row['NE-FNR']
    #     print(f'& {wer} \\,\\textbar\, {ne_wer} \\,\\textbar\\, {ne_fnr}')
    #     if (i + 1)  == 4:
    #         print('\\\\')
    #         print('&  & Coarse')
    #     if (i + 1)  == 8:
    #         print('\\\\')
    #         print('&  & Fine')
    #     if (i + 1)  == 12:
    #         print('\\\\')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python view_metric_result.py <input_file> <output_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_excel_base = sys.argv[2]
    aggregate_metrics(input_path, output_excel_base)
