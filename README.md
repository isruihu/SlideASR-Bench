# Look before Transcription: End-to-End SlideASR with Visually-Anchored Policy Optimization
<p align="center" dir="auto">
<a href="https://arxiv.org/abs/2510.08618" rel="nofollow"><img src="https://img.shields.io/badge/ArXiv-2510.08618-red" style="max-width: 100%;"></a>
<a href="https://huggingface.co/datasets/RUIH/SlideASR-Bench" rel="nofollow"><img src="https://img.shields.io/badge/Dataset-SlideASR_Bench-yellow" style="max-width: 100%;"></a>
<a href="https://huggingface.co/datasets/RUIH/VAPO-7B" rel="nofollow"><img src="https://img.shields.io/badge/Model-VAPO-blue" style="max-width: 100%;"></a>
</p>

Automatic speech recognition (ASR) systems often struggle with domain-specific terminology, especially in specialized settings such as academic lectures. To address this, we define the SlideASR task, which leverages the rich visual information from presentation slides to improve transcription accuracy. Existing pipeline methods for this task tend to be complex and underperform. Although omni-modal large language models (OLLMs) provide a promising end-to-end framework, they frequently fail in practice by degenerating into simple optical character recognition (OCR) systems. To overcome this, we propose Visually-Anchored Policy Optimization (VAPO), a novel post-training method designed to control the model's reasoning process. Drawing on the Chain-of-Thought reasoning paradigm, VAPO enforces a structured "Look before Transcription" procedure using a <think><answer> format. Specifically, the model first performs OCR on the slide content within the think step, then generates the transcription by referencing this recognized visual information in the answer step. This reasoning process is optimized via reinforcement learning with four distinct rewards targeting format compliance, OCR accuracy, ASR quality, and visual anchoring consistency. To support further research, we construct SlideASR-Bench, a new entity-rich benchmark consisting of a synthetic dataset for training and testing, and a challenging real-world set for evaluation. Extensive experiments demonstrate that VAPO significantly improves recognition of domain-specific terms, establishing an effective end-to-end paradigm for SlideASR.

## Download SlideASR-Bench Data  
The SildeASR-Bench dataset is now available for download at [🤗Huggingface🤗](https://huggingface.co/datasets/RUIH/SlideASR-Bench). 

## Model

The VAPO models are now available for download.

| Model | Download URL |
| --- | --- |
| VAPO-3B | [🤗Huggingface🤗](https://huggingface.co/RUIH/VAPO-3B) |
| VAPO-7B | [🤗Huggingface🤗](https://huggingface.co/RUIH/VAPO-7B) |

## Get Results on SlideASR-Bench

### Environment
```shell
pip install -r requirements.txt
```
### Setup
Modify the model path in the config.py file.

### Run
```shell
CUDA_VISIBLE_DEVICES=0,1 python run.py --model {MODEL_NAME} --dataset SlideASR-S --models-per-gpu {MODELS_PER_GPU}
```

## Evaluation
```shell
bash slide_asr/SlideASR-Bench/evaluation/evaluate.sh {MODEL_NAME}
```

## Acknowledgement
> We sincerely thank [ContextASR-Bench](https://github.com/MrSupW/ContextASR-Bench) for providing the datasets.


## 📚 Citation
```
@article{hu2025vapo,
      title={Look before Transcription: End-to-End SlideASR with Visually-Anchored Policy Optimization}, 
      author={Rui Hu and Delai Qiu and Yining Wang and Shengping Liu and Jitao Sang},
      year={2025},
      eprint={2510.08618},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2510.08618}, 
}
```
