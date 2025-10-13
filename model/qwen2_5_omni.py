from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniThinkerForConditionalGeneration
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
from transformers import Qwen2_5OmniProcessor, AutoTokenizer, AutoConfig
from qwen_omni_utils import process_mm_info

import torch
import torch.multiprocessing as mp
import json
import os
import copy
from tqdm import tqdm
from loguru import logger as logu
from .base_model import BaseModel, BaseAVModel


MAX_PIXELS = 128 * 5 * 72 * 5


# Audio-only
class Qwen2_5OmniWorker(BaseModel):
    """
    一个封装了模型加载和推理逻辑的工作类。
    每个实例将在一个独立的进程中被创建和使用。
    """
    MAX_SUPPORTED_SECONDS = 180
    def __init__(self, model_path: str, device: str):
        logu.info(f"工作进程 {os.getpid()} 正在 GPU {device} 上初始化 InferenceWorker...")
        self.device = device
        self.model_path = model_path
        
        AutoConfig.register("qwen2_5_omni_thinker", Qwen2_5OmniThinkerConfig)
        config = AutoConfig.from_pretrained(self.model_path)
        
        model_class = Qwen2_5OmniForConditionalGeneration if config.model_type == 'qwen2_5_omni' else Qwen2_5OmniThinkerForConditionalGeneration
        loaded_model = model_class.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2"
        )
        if hasattr(loaded_model, 'thinker') and config.model_type == 'qwen2_5_omni':
            self.model = loaded_model.thinker
            if not hasattr(self.model, 'generate'):
                 self.model = loaded_model
        else:
            self.model = loaded_model

        if 'enable_audio_output' in self.model.config.to_dict():
            self.model.config.enable_audio_output = False
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        logu.info(f"工作进程 {os.getpid()} 初始化完成。")

    def omni_run(self, conversation):
        USE_AUDIO_IN_VIDEO = False
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(self.device, dtype=self.model.dtype)
        with torch.no_grad():
            text_ids = self.model.generate(
                return_dict_in_generate=True,
                max_new_tokens=1024,
                do_sample=False,
                **inputs,
            ).sequences
        decoded_text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = decoded_text[0].split("\nassistant\n")[-1]
        return response

    def run_atom(self, audio, prompt):
        """为单个音频和提示语执行推理。"""
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a speech recognition model."}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio}, {"type": "text", "text": prompt}]},
        ]
        return self.omni_run(conversation)


# Image + Audio
class Qwen2_5OmniAVWorker(BaseAVModel, Qwen2_5OmniWorker):
    def __init__(self, model_path: str, device: str):
        Qwen2_5OmniWorker.__init__(self, model_path, device)

    def run_atom(self, image, audio, prompt):
        """为单个音频和提示语执行推理。"""
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a speech recognition model."}]},
            {"role": "user", "content": [
                {"type": "image", "image": image, "max_pixels": MAX_PIXELS},
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt}
            ]},
        ]
        return self.omni_run(conversation)


# Ours
class VAPOWorker(BaseAVModel, Qwen2_5OmniWorker):
    def __init__(self, model_path: str, device: str):
        Qwen2_5OmniWorker.__init__(self, model_path, device)

    def run_atom(self, image, audio, prompt):
        """为单个音频和提示语执行推理。"""
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Your task is to convert the speech into text, and the image serves as the reference content related to the speech."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "image", "image": image, 'max_pixels': MAX_PIXELS},
                    {"type": "text", "text": "First, recognize the text in the image and output it within <think> </think>. Then, referring to the thinking content, output the speech recognition result within <answer> </answer>"}
                ]
            }
        ]
        return self.omni_run(conversation)

