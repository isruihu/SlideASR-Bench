from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

import torch
import os
from loguru import logger as logu

from .base_model import BaseModel, BaseAudioModelWithOCR


class MiDashengWorker(BaseModel):
    """
    一个封装了模型加载和推理逻辑的工作类。
    每个实例将在一个独立的进程中被创建和使用。
    """
    def __init__(self, model_path: str, device: str):
        logu.info(f"工作进程 {os.getpid()} 正在 GPU {device} 上初始化 InferenceWorker...")
        self.device = device
        self.model_path = model_path
        
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map=device, 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        )    
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

        self.MAX_SUPPORTED_SECONDS = 60
        logu.info(f"工作进程 {os.getpid()} 初始化完成。")

    def run_atom(self, audio, prompt):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful language and speech assistant."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "audio",
                        "path": audio
                    },
                ],
            },
        ]
        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=True,
            return_dict=True,
        )
        model_inputs = model_inputs.to(self.device, dtype=self.model.dtype)
        generation = self.model.generate(**model_inputs)
        response = self.tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
        return response


class MiDashengWithOCRWorker(BaseAudioModelWithOCR, MiDashengWorker):
    def __init__(self, model_path: str, device: str):
        BaseAudioModelWithOCR.__init__(self, model_path, device)
        MiDashengWorker.__init__(self, model_path, device)
        
    def run_atom(self, audio, prompt):
        logu.info(prompt)
        return MiDashengWorker.run_atom(self, audio, prompt)
