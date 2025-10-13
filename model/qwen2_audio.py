from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, AutoTokenizer

import librosa
import torch
import json
import os
import copy
from tqdm import tqdm
from loguru import logger as logu
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import re

from .base_model import BaseModel, BaseAudioModelWithOCR


class Qwen2AudioWorker(BaseModel):
    """
    一个封装了模型加载和推理逻辑的工作类。
    每个实例将在一个独立的进程中被创建和使用。
    """
    def __init__(self, model_path: str, device: str):
        logu.info(f"工作进程 {os.getpid()} 正在 GPU {device} 上初始化 InferenceWorker...")
        self.device = device
        self.model_path = model_path
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16)    
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

        self.MAX_SUPPORTED_SECONDS = 30
        logu.info(f"工作进程 {os.getpid()} 初始化完成。")

    def run_atom(self, audio, prompt):
        conversation = [{
            "role": "user", "content": [
                {"type": "audio", "audio_url": audio},
                {"type": "text", "text": prompt},
            ]
        }]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(
                            librosa.load(ele['audio_url'], sr=self.processor.feature_extractor.sampling_rate)[0]
                        )
        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        generate_ids = self.model.generate(**inputs, max_length=1024)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = self.extract_text_between_quotes(response)
        return response

    def extract_text_between_quotes(self, text):
        """
        判断字符串中是否有冒号，如果有，提取第一个和最后一个单引号之间的内容。

        Args:
            text: 输入的字符串。

        Returns:
            如果找到冒号且字符串中至少有两个单引号，则返回提取的内容。
            否则，返回 None。
        """
        if ':' in text or '：' in text:
            try:
                # 查找第一个单引号的位置
                first_quote_index = text.find("'")
                # 查找最后一个单引号的位置
                last_quote_index = text.rfind("'")

                # 确保两个单引号都存在，并且第一个在最后一个前面
                if first_quote_index != -1 and last_quote_index != -1 and last_quote_index > first_quote_index:
                    # 提取并返回两个单引号之间的内容
                    return text[first_quote_index + 1 : last_quote_index]
            except Exception as e:
                return text
        return text


class Qwen2AudioWithOCRWorker(BaseAudioModelWithOCR, Qwen2AudioWorker):
    def __init__(self, model_path: str, device: str):
        BaseAudioModelWithOCR.__init__(self, model_path, device)
        Qwen2AudioWorker.__init__(self, model_path, device)
        
    def run_atom(self, audio, prompt):
        logu.info(prompt)
        return Qwen2AudioWorker.run_atom(self, audio, prompt)

