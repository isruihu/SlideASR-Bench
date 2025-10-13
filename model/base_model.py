import numpy as np
import tempfile
import os
from pathlib import Path
from loguru import logger as  logu
from pydub import AudioSegment
from pydub.silence import split_on_silence

MAX_CHUNK_DURATION_MS = 29500 
MIN_SILENCE_LEN_MS = 500
KEEP_SILENCE_MS = 300


PADDLE_ROOT = Path(__file__).parent.parent / "resource" / "PaddlePaddle"
det_model_dir = os.path.join(PADDLE_ROOT, 'PP-OCRv5_mobile_det')
rec_model_dir = os.path.join(PADDLE_ROOT, 'PP-OCRv5_mobile_rec')

class BaseModel():
    """
    输入为Audio+Prompt
    """
    
    SUPPORT_IMAGE = False

    def __init__(self, model_path, device):
        self.model = None
        self.MAX_SUPPORTED_SECONDS = float('inf')

    def run_inference(self, audio_url: str, prompt: str) -> str:
        audio = AudioSegment.from_file(audio_url)
        duration = len(audio) / 1000.0
        # 时长在模型支持范围内
        if duration < self.MAX_SUPPORTED_SECONDS:
            asr_text = self.run_atom(audio_url, prompt)
        else:
            asr_text = self.run_inference_with_splitting(long_audio=audio, prompt=prompt)
        # logu.info(f'ASR text: {asr_text}')
        return asr_text

    def run_atom(self, audio: str | np.ndarray, prompt: str) -> str:
        pass

    def run_inference_with_splitting(self, long_audio, prompt: str) -> str:
        """
        加载音频，按静音切分，然后对每个块进行推理，最后合并结果。
        """
        # 2. 按静音切分音频
        silence_thresh = long_audio.dBFS - 16 # 根据音频平均响度动态设置静音阈值
        chunks = split_on_silence(
            long_audio,
            min_silence_len=MIN_SILENCE_LEN_MS,
            silence_thresh=silence_thresh,
            keep_silence=KEEP_SILENCE_MS
        )
        
        # 如果未检测到静音，则将整个音频视为一个块
        chunks = [chunk for chunk in chunks if len(chunk) != 0]
        logu.info(f"chunk_size: {len(chunks)}: {[len(c) for c in chunks]}")

        # 3. 合并过短的音频块，确保每个块都有效利用模型，同时不超过最大长度
        final_chunks = []
        current_chunk = AudioSegment.empty()
        for chunk in chunks:
            if len(current_chunk) + len(chunk) < MAX_CHUNK_DURATION_MS:
                current_chunk += chunk
            else:
                if len(current_chunk) > 0:
                    final_chunks.append(current_chunk)
                current_chunk = chunk
        if len(current_chunk) > 0:
            final_chunks.append(current_chunk)
        
        logu.info(f"chunk_size: {len(final_chunks)}: {[len(c) for c in final_chunks]}")

        audio_temp_files = []
        full_transcript = []

        try:
            for i, chunk in enumerate(final_chunks):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_f:
                    temp_filename = temp_f.name
                    chunk.export(temp_filename, format="wav")
                    audio_temp_files.append(temp_filename)
            
            for f_path in audio_temp_files:
                text_part = self.run_atom(audio=f_path, prompt=prompt)
                full_transcript.append(text_part.strip())

        finally:
            for f_path in audio_temp_files:
                os.remove(f_path)

        # 5. 合并所有识别结果
        return "\n".join(full_transcript)

class BaseAVModel():
    """
    输入为Image+Audio+Prompt
    """

    SUPPORT_IMAGE = True
    
    def __init__(self, model_path, device):
        self.model = None
        self.MAX_SUPPORTED_SECONDS = float('inf')

    def run_inference(self, image_url: str, audio_url: str, prompt: str) -> str:
        """
        image_url: 图像路径
        audio_url: 音频路径
        prompt: 用户指令
        """
        audio = AudioSegment.from_file(audio_url)
        duration = len(audio) / 1000.0
        # 时长在模型支持范围内
        if duration < self.MAX_SUPPORTED_SECONDS:
            asr_text = self.run_atom(image_url, audio_url, prompt)
        else:
            raise NotImplementedError(f"音频长度{duration}超出模型支持最大长度{self.MAX_SUPPORTED_SECONDS}")
        # logu.info(f'ASR text: {asr_text}')
        return asr_text

    def run_atom(self, image: str, audio: str, prompt: str) -> str:
        pass

class BaseAudioModelWithOCR(BaseModel):
    """
    输入为Image+Audio+Prompt
    """
    SUPPORT_IMAGE = True
    USE_OCR = True

    def __init__(self, model_path, device):
        self.model = None
        self.MAX_SUPPORTED_SECONDS = float('inf')
        from paddleocr import PaddleOCR
        self.ocr_model = PaddleOCR(
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            text_detection_model_name="PP-OCRv5_mobile_det",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir
        )

    def get_ocr_text(self, image):
        result = self.ocr_model.predict(input=image)
        rec_texts = result[0]['rec_texts']
        rec_texts = ' '.join(rec_texts)
        return rec_texts

    def run_inference(self, image_url: str, audio_url: str, prompt: str) -> str:
        """
        image_url: 图像路径
        audio_url: 音频路径
        prompt: 用户指令
        """
        # update prompt
        ocr_text = self.get_ocr_text(image_url)
        prompt = prompt.format(ocr_text)
        return BaseModel.run_inference(self, audio_url=audio_url, prompt=prompt)
