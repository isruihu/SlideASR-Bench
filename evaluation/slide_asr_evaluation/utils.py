
import math
import regex as re
import contractions
import editdistance

from collections import defaultdict


EN_PUNCS_MID_STOP = re.escape(',;(){}[]"|:')
EN_PUNCS_END_STOP = re.escape('!?.')
EN_PUNCS_NON_STOP = re.escape('#$%&*+/<=>@\\^_`~"')  # without "'" and "-"
CN_PUNCS_MID_STOP = re.escape('，；､、丶｟｠《》（）｢｣［］｛｝「｣『』【】〔〕〖〗〘〙〚〛〈〉｜：：')
CN_PUNCS_END_STOP = re.escape('！？｡。')
CN_PUNCS_NON_STOP = re.escape('＂＃＄％＆＇＊＋－／＜＝＞＠＼＾＿｀～〃〜〝〞〟〰〾〿‘’‛“”„‟…‧﹏·•・′″–—―')
all_punctuations = EN_PUNCS_MID_STOP + EN_PUNCS_END_STOP + EN_PUNCS_NON_STOP + CN_PUNCS_MID_STOP + CN_PUNCS_END_STOP + CN_PUNCS_NON_STOP


def extract_entities(text, entities_list, entity2count=None):
    """
    Extract all entities that appear in the text precisely in order.
    Args:
        text (str): The text to extract entities from.
        entities_list (list): A list of entities to extract.
        entity2count (dict): A dictionary of entity counts.
    Returns:
        ordered_entities (list): A list of extracted entities in order.
    """
    text_tokens = text.split()
    entities_with_tokens = []
    for entity in entities_list:
        tokens = entity.split()
        entities_with_tokens.append((tokens, entity))

    match_entity2count = defaultdict(int)
    # Store matches with their start position and length
    matches = []
    n = len(text_tokens)

    for i in range(n):
        for (e_tokens, e_str) in entities_with_tokens:
            l = len(e_tokens)
            if i + l > n:
                continue
            if text_tokens[i:i + l] == e_tokens:
                # Prevent the hallucinations in the identification results from repeatedly interfering with the extraction results
                if entity2count and match_entity2count[e_str] >= entity2count[e_str]:
                    continue
                # Record start position and length for sorting
                match_entity2count[e_str] += 1
                matches.append((i, l, e_str))
    # Sort matches by their occurrence position and entity length (shorter first)
    matches.sort(key=lambda x: (x[0], x[1]))
    ordered_entities = [entity for (_, _, entity) in matches]
    return ordered_entities


def extract_entities_fuzzy(text, entities_list):
    """
    Vaguely match all the entities that may be correct in the text in order.
    Args:
        text (str): Text to extract entities from.
        entities_list (list): List of entities to match.
    Returns:
        ordered_entities (list): A list of fuzzy matched entities in order.
    """
    text_tokens = text.split()
    # Store matches with their start positions
    match_positions = []
    for entity in entities_list:
        entity_tokens = entity.split()
        n = len(entity_tokens)
        if n == 0:
            continue
        max_dist = math.ceil(n / 2) - 1
        min_len = max(1, n - max_dist)
        max_len = n + max_dist
        lengths_to_search = [n] + list(range(n - 1, min_len - 1, -1)) + list(range(n + 1, max_len + 1))
        # Scan through all possible windows
        next_start = 0
        for start in range(len(text_tokens)):
            if start < next_start:
                continue
            for length in lengths_to_search:
                end = start + length
                if end > len(text_tokens):
                    break
                window = text_tokens[start:end]
                distance = editdistance.eval(window, entity_tokens)
                if distance <= max_dist:
                    next_start = end
                    window_text = ' '.join(window)
                    # Record match with its start position
                    search = re.search(re.escape(entity), window_text)
                    if search:
                        matched_entity = entity
                        next_start -= len(window_text[search.end():].strip().split())
                    else:
                        matched_entity = window_text
                    match_positions.append((start, matched_entity))
                    break
    # Sort matches by their occurrence position and entity length (shorter first)
    match_positions.sort(key=lambda x: (x[0], len(x[1].split())))
    # Extract sorted entities while preserving original text order
    seen = set()
    ordered_entities = []
    for pos, entity in match_positions:
        if (pos, entity) not in seen:
            seen.add((pos, entity))
            ordered_entities.append(entity)
    return ordered_entities


def calculate_wer(asr_text, target_text):
    """
    Calculate the Word Error Rate (WER) between two texts.
    Args:
        asr_text (str): The ASR output text.
        target_text (str): The ground truth text.
    Returns:
        wer (float): The WER score.
        I (int): Number of insertions.
        D (int): Number of deletions.
        S (int): Number of substitutions.
    """
    dp = [[0] * (len(target_text) + 1) for _ in range(len(asr_text) + 1)]
    insertions = [[0] * (len(target_text) + 1) for _ in range(len(asr_text) + 1)]
    deletions = [[0] * (len(target_text) + 1) for _ in range(len(asr_text) + 1)]
    substitutions = [[0] * (len(target_text) + 1) for _ in range(len(asr_text) + 1)]

    for i in range(1, len(asr_text) + 1):
        dp[i][0] = i
        insertions[i][0] = i

    for j in range(1, len(target_text) + 1):
        dp[0][j] = j
        deletions[0][j] = j

    for i in range(1, len(asr_text) + 1):
        for j in range(1, len(target_text) + 1):
            if asr_text[i - 1] == target_text[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 无操作
                insertions[i][j] = insertions[i - 1][j - 1]
                deletions[i][j] = deletions[i - 1][j - 1]
                substitutions[i][j] = substitutions[i - 1][j - 1]
            else:
                insertion = dp[i - 1][j] + 1
                deletion = dp[i][j - 1] + 1
                substitution = dp[i - 1][j - 1] + 1
                dp[i][j] = min(insertion, deletion, substitution)
                if dp[i][j] == substitution:
                    insertions[i][j] = insertions[i - 1][j - 1]
                    deletions[i][j] = deletions[i - 1][j - 1]
                    substitutions[i][j] = substitutions[i - 1][j - 1] + 1
                elif dp[i][j] == deletion:
                    insertions[i][j] = insertions[i][j - 1]
                    deletions[i][j] = deletions[i][j - 1] + 1
                    substitutions[i][j] = substitutions[i][j - 1]
                else:
                    insertions[i][j] = insertions[i - 1][j] + 1
                    deletions[i][j] = deletions[i - 1][j]
                    substitutions[i][j] = substitutions[i - 1][j]

    wer = dp[len(asr_text)][len(target_text)] / len(target_text) if len(target_text) > 0 else 0
    I = insertions[len(asr_text)][len(target_text)]
    D = deletions[len(asr_text)][len(target_text)]
    S = substitutions[len(asr_text)][len(target_text)]

    return wer, I, D, S


def merge_single_letters(s):
    """
    Combine adjacent single letters separated by spaces.
    Args:
        s (str): input string
    Returns:
        str: merged string
    """
    words = s.split()
    current = []
    result = []
    for word in words:
        if not word:  # 处理空字符串的情况
            if current:
                result.append(''.join(current))
                current = []
            result.append(word)
            continue
        first_char = word[0]
        remaining = word[1:] if len(word) > 1 else ''
        if (first_char.islower() or first_char.isupper()):
            valid_remaining = (remaining == 's' or remaining == "'s")
            if remaining == '' or valid_remaining:
                current.append(first_char)
                if remaining:
                    current.append(remaining)
            else:
                if current:
                    result.append(''.join(current))
                    current = []
                result.append(word)
        else:
            if current:
                result.append(''.join(current))
                current = []
            result.append(word)
    if current:
        result.append(''.join(current))
    return ' '.join(result)


def simple_tokenize(text, lang="zh"):
    """
    Simple text tokenization, character for Chinese and word for English.
    Args:
        text (str): input text
        lang (str): language of the text
    Return:
        str: normalized text
    """
    if text.isupper():
        text = text.lower()
    if lang == "en":
        text = re.sub(r"^(O')\s|\s(O')$|\s(O')\s", ' O ', text)
        text = re.sub(r"^(o')\s|\s(o')$|\s(o')\s", ' o ', text)
        text = contractions.fix(text, leftovers=False, slang=False)
    text = re.sub(rf"[{all_punctuations}]", ' ', text)
    text = text.replace('-', ' ')
    text = text.replace('\'', ' ')
    ckj_characters = r"\p{Han}\p{Hangul}\p{Hiragana}\p{Katakana}"
    latin_characters = r"\p{IsLatin}"
    text = re.sub(rf'(?<=[{ckj_characters}])(?=[{ckj_characters}])', ' ', text)
    text = re.sub(rf'(?<=[{ckj_characters}])(?={latin_characters})', ' ', text)
    text = re.sub(rf'(?<={latin_characters})(?=[{ckj_characters}])', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = merge_single_letters(text)
    return text.lower()
