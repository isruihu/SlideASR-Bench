import sys
import json
import concurrent.futures

from tqdm import tqdm
from collections import defaultdict
from utils import simple_tokenize, extract_entities, extract_entities_fuzzy
import re


def process_line(line):
    json_data = json.loads(line.strip())
    asr_info = {}
    lang = "zh" if json_data["language"] == "Chinese" or json_data["language"] == "Mandarin" else "en"
    type_ = "dialogue" if "dialogue" in json_data.keys() else "speech"
    json_data ["type"] = type_
    
    json_data["norm_text"] = simple_tokenize(json_data["text"], lang)
    target_text = json_data["norm_text"]
    entity_list = json_data["entity_list"]
    norm_entity_list = []
    for entity in entity_list:
        norm_entity = simple_tokenize(entity, lang)
        if not norm_entity in target_text:
            print(f"{json_data['uniq_id']} {entity} -> {norm_entity} {target_text}")
            return None
        norm_entity_list.append(norm_entity)
    json_data["norm_entity_list"] = norm_entity_list
    json_data["match_entities"] = extract_entities(target_text, norm_entity_list)
    entity2count = defaultdict(int)
    for entity in json_data["match_entities"]:
        entity2count[entity] += 1

    for asr_model, asr_result in json_data["asr_info"].items():
        asr_text = asr_result["asr_text"]
        answer = re.findall(r'<answer>.*?</answer>', asr_text, re.DOTALL)
        if len(answer) != 0:
            asr_text = answer[0]
        else:
            if '<think>' in asr_text:
                asr_text = ''

        norm_asr_text = simple_tokenize(asr_text, lang)
        match_entities = extract_entities(norm_asr_text, norm_entity_list, entity2count)
        fuzzy_match_entities = extract_entities_fuzzy(norm_asr_text, norm_entity_list)
        fuzzy_match_entities_set = []
        
        # 去重
        for e in fuzzy_match_entities:
            if e not in fuzzy_match_entities_set:
                fuzzy_match_entities_set.append(e)

        asr_result.update({
            "norm_asr_text": norm_asr_text,
            "match_entities": match_entities,
            "fuzzy_match_entities": fuzzy_match_entities_set
        })
        asr_info[asr_model] = asr_result
    del json_data["asr_info"]
    json_data["asr_info"] = asr_info
    return json.dumps(json_data, ensure_ascii=False) + '\n'


if __name__ == "__main__":
    assert len(sys.argv) == 3, "python3 caculate_metrics.py <input_jsonl> <output_jsonl>"
    input_jsonl, output_jsonl = sys.argv[1], sys.argv[2]

    lines = open(input_jsonl, 'r').readlines()

    failed_cnt = 0
    prepared_data_lines = []
    # Create process pool and process lines in parallel
    pbar = tqdm(total=len(lines), desc="Processing lines")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(process_line, lines):
            if result is not None:
                prepared_data_lines.append(result)
            else:
                failed_cnt += 1
            pbar.update(1)
    pbar.close()
    print(f"Failed cnt: {failed_cnt}")
    # Write results to output file
    with open(output_jsonl, 'w') as f:
        f.writelines(prepared_data_lines)
