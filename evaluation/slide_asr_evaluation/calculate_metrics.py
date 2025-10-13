import sys
import json
import concurrent.futures

from tqdm import tqdm
from utils import calculate_wer


def process_line(line):
    json_data = json.loads(line.strip())
    target_text = json_data["norm_text"]
    target_entity_list = json_data["match_entities"]
    target_entity_text = ' '.join(target_entity_list)

    T = len(target_text.split())
    ne_T = len(target_entity_list)
    ne_text_T = len(target_entity_text.split())

    for asr_model, asr_result in json_data["asr_info"].items():
        norm_asr_text = asr_result["norm_asr_text"]
        wer, I, D, S = calculate_wer(norm_asr_text.split(), target_text.split())
        assert abs((I + D + S) / T - wer) < 1e-5, f"{I} + {D} + {S} / {T} != {wer}"
        asr_entity_list = asr_result["match_entities"]
        asr_fuzzy_entity_list = asr_result["fuzzy_match_entities"]
        ne_wer, ne_I, ne_D, ne_S = calculate_wer(' '.join(asr_fuzzy_entity_list).split(), target_entity_text.split())
        assert abs((ne_I + ne_D + ne_S) / ne_text_T - ne_wer) < 1e-5, f"{ne_I} + {ne_D} + {ne_S} / {ne_text_T} != {ne_wer}"
        json_data["asr_info"][asr_model].update({
            "wer_info": {
                "wer": wer,
                "I": I,
                "D": D,
                "S": S,
                "T": T
            },
            "ne_wer_info": {
                "wer": ne_wer,
                "I": ne_I,
                "D": ne_D,
                "S": ne_S,
                "T": ne_text_T
            },
            "ne_fnr_info": {
                "fnr": 1 - len(asr_entity_list) / ne_T,
                "H": len(asr_entity_list),
                "T": ne_T
            }
        })

    # return json.dumps(json_data, ensure_ascii=False) + '\n'
    return json_data


if __name__ == "__main__":
    assert len(sys.argv) == 3, "python3 caculate_metrics.py <input_jsonl> <output_jsonl>"
    input_jsonl, output_jsonl = sys.argv[1], sys.argv[2]

    lines = open(input_jsonl, 'r').readlines()

    calculated_data_lines = []
    # Create process pool and process lines in parallel
    pbar = tqdm(total=len(lines), desc="Processing lines")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(process_line, lines):
            calculated_data_lines.append(result)
            pbar.update(1)
    pbar.close()

    key = list(calculated_data_lines[0]['asr_info'].keys())[0]
    # calculated_data_lines.sort(key=lambda x: x['asr_info'][key]['wer_info']['wer'], reverse=True)
    calculated_data_lines.sort(key=lambda x: x['asr_info'][key]['ne_wer_info']['wer'], reverse=True)
    # Write results to output file
    with open(output_jsonl, 'w') as f:
        for line in calculated_data_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
