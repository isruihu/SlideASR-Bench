import os
import json
from torch.utils.data import Dataset
from loguru import logger as logu
from pathlib import Path


current_dir = Path(__file__).parent
ROOT_SLIDE_ASR_BENCH = current_dir.parent / "resource" / "SlideASR-Bench"


class SlideASR_S(Dataset):
    ROOT = os.path.join(ROOT_SLIDE_ASR_BENCH, 'SlideASR-S')
    data_file = os.path.join(ROOT_SLIDE_ASR_BENCH, 'SlideASR-S/test.jsonl')

    def __init__(self, args, model_class):
        # --- 数据加载和准备（主进程） ---
        data = [json.loads(it) for it in open(self.data_file).readlines()]
        saved_ids = set()
        if os.path.exists(args.output_file):
            with open(args.output_file, 'r') as fr:
                for line in fr:
                    try:
                        item = json.loads(line)
                        saved_ids.add(item['uniq_id'])
                    except json.JSONDecodeError:
                        logu.warning(f"无法解析已存在输出文件中的一行: {line}")
    
        filtered_data = [item for item in data if item.get('uniq_id') not in saved_ids]
        logu.info(f"共找到 {len(data)} 个样本。{len(saved_ids)} 个已处理。将处理 {len(filtered_data)} 个新样本。")        
        self.filtered_data = filtered_data

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        return self.filtered_data[idx]

    @classmethod
    def get_prompt(cls, model, item: dict) -> str:
        if item['language'] == 'Mandarin':
            if model.SUPPORT_IMAGE:
                if hasattr(model, 'USE_OCR') and model.USE_OCR:
                    prompt = "语音为演讲者结合PPT的发言，PPT文字为：\n{}\n结合PPT内容，将语音转录为文本。"
                else:
                    prompt = "结合图像，将语音转成文本。"
            else:
                prompt = '将语音转成文本。'
        else:
            if model.SUPPORT_IMAGE:
                if hasattr(model, 'USE_OCR') and model.USE_OCR:
                    prompt = "The speech is the speaker's talk accompanied by a slide, with the text of the slide being:\n{}\nTranscribe the speech into text by integrating the speech with the slide content."
                else:
                    prompt = 'Transcribe the speech into text by integrating the speech with the slide content.'
            else:
                prompt = 'Transcribe the speech into text.'

        return prompt

    @classmethod
    def process_item(cls, model, item: dict) -> dict:
        """工作函数，处理单个数据项。"""
        audio_url = os.path.join(cls.ROOT, item['audio'])
        image_url = os.path.join(cls.ROOT, item['slide'])
        prompt = cls.get_prompt(model, item)
        
        try:
            if model.SUPPORT_IMAGE:
                context = 'image+audio'
                asr_text = model.run_inference(image_url=image_url, audio_url=audio_url, prompt=prompt)
            else:
                context = 'audio'
                asr_text = model.run_inference(audio_url=audio_url, prompt=prompt)
            item['asr_info'] = {context: {}}
            item['asr_info'][context]['prompt'] = prompt
            item['asr_info'][context]['asr_text'] = asr_text
            return item
        except Exception as e:
            logu.error(f"处理项目 {item['uniq_id']} 时, 发生错误: {e}")
            logu.error(f"audio_url: {audio_url}")
            item['error'] = str(e)
            return item


class SlideASR_R(SlideASR_S):
    ROOT = os.path.join(ROOT_SLIDE_ASR_BENCH, 'SlideASR-R')
    data_file = os.path.join(ROOT_SLIDE_ASR_BENCH, 'SlideASR-R/test.jsonl')
