import soundfile as sf
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

import torch
import torch.multiprocessing as mp
import json
import os
import copy
from tqdm import tqdm
from loguru import logger as logu
from .base_model import BaseModel, BaseAVModel, BaseAudioModelWithOCR


MAX_PIXELS = 128 * 5 * 72 * 5


class Qwen3OmniWorker(BaseModel):

    MAX_SUPPORTED_SECONDS = 180
    
    def __init__(self, model_path: str, device: str):
        logu.info(f"工作进程 {os.getpid()} 正在 GPU {device} 上初始化 InferenceWorker...")
        self.device = device
        self.model_path = model_path

        model_class = Qwen3OmniMoeForConditionalGeneration
        loaded_model = model_class.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2"
        )
        self.model = loaded_model

        if 'enable_audio_output' in self.model.config.to_dict():
            self.model.config.enable_audio_output = False
        self.model.eval()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path,)
        logu.info(f"工作进程 {os.getpid()} 初始化完成。")

    def omni_run(self, messages):
        use_audio_in_video = False
        return_audio = False
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        text_ids, audio = self.model.generate(**inputs, 
                                            thinker_return_dict_in_generate=True,
                                            thinker_max_new_tokens=8192, 
                                            thinker_do_sample=True,
                                            speaker="Ethan", 
                                            use_audio_in_video=use_audio_in_video,
                                            return_audio=return_audio)
        response = self.processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response

    def run_atom(self, audio, prompt):
        """为单个音频和提示语执行推理。"""
        conversation = [
            {"role": "user", "content": [{"type": "audio", "audio": audio}, {"type": "text", "text": prompt}]},
        ]
        return self.omni_run(conversation)


class Qwen3OmniAVWorker(BaseAVModel, Qwen3OmniWorker):
    def __init__(self, model_path: str, device: str):
        Qwen3OmniWorker.__init__(self, model_path, device)

    def run_atom(self, image, audio, prompt):
        """为单个音频和提示语执行推理。"""
        conversation = [
            {"role": "user", "content": [
                {"type": "image", "image": image, "max_pixels": MAX_PIXELS},
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt}
            ]},
        ]
        return self.omni_run(conversation)


class Qwen3OmniWithOCRWorker(BaseAudioModelWithOCR, Qwen3OmniWorker):
    def __init__(self, model_path: str, device: str):
        BaseAudioModelWithOCR.__init__(self, model_path, device)
        Qwen3OmniWorker.__init__(self, model_path, device)
        
    def run_atom(self, audio, prompt):
        logu.info(prompt)
        return Qwen3OmniWorker.run_atom(self, audio, prompt)

