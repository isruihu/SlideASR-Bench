# 导入所有必要的模块和类
from model import (
    Qwen2_5OmniWorker, Qwen2_5OmniAVWorker, VAPOWorker,
    Qwen3OmniWorker, Qwen3OmniAVWorker, Qwen3OmniWithOCRWorker,
    Qwen2AudioWorker, Qwen2AudioWithOCRWorker,
    MiDashengWorker, MiDashengWithOCRWorker,
)
from dataset import (
    SlideASR_S, SlideASR_R
)

HF_DIR = '/path/to/your/huggingface/dir'

DATASET_REGISTRY = {
    'SlideASR-S': SlideASR_S, 'SlideASR-R': SlideASR_R
}

AUDIO_MODEL = {
    'qwen-omni': (Qwen2_5OmniWorker, f'{HF_DIR}/Qwen/Qwen2.5-Omni-3B/'),
    'qwen-omni-7b': (Qwen2_5OmniWorker, f'{HF_DIR}/Qwen/Qwen2.5-Omni-7B/'),
    'qwen3-omni': (Qwen3OmniWorker, f'{HF_DIR}/Qwen/Qwen3-Omni-30B-A3B-Instruct'),
    'qwen2-audio': (Qwen2AudioWorker, f'{HF_DIR}/Qwen/Qwen2-Audio-7B-Instruct/'),
    'mi-dasheng': (MiDashengWorker, f'{HF_DIR}/mispeech/midashenglm-7b'),
}

AUDIO_IMAGE_MODEL = {
    # Omni-LLM
    'qwen-omni-av': (Qwen2_5OmniAVWorker, f'{HF_DIR}/Qwen/Qwen2.5-Omni-3B/'),
    'qwen-omni-av-7b': (Qwen2_5OmniAVWorker, f'{HF_DIR}/Qwen/Qwen2.5-Omni-7B/'),
    'qwen3-omni-av': (Qwen3OmniAVWorker, f'{HF_DIR}/Qwen/Qwen3-Omni-30B-A3B-Instruct'),
    # Audio-LLM + OCR
    'qwen2-audio-ocr': (Qwen2AudioWithOCRWorker, f'{HF_DIR}/Qwen/Qwen2-Audio-7B-Instruct/'),
    'mi-dasheng-ocr': (MiDashengWithOCRWorker, f'{HF_DIR}/mispeech/midashenglm-7b'),
    'qwen3-omni-ocr': (Qwen3OmniWithOCRWorker, f'{HF_DIR}/Qwen/Qwen3-Omni-30B-A3B-Instruct'),
    # VAPO model
    'VAPO-3B': (VAPOWorker, f'{HF_DIR}/RUIH/VAPO-3B'),
    'VAPO-7B': (VAPOWorker, f'{HF_DIR}/RUIH/VAPO-7B'),
}

MODEL_REGISTRY = AUDIO_MODEL | AUDIO_IMAGE_MODEL
