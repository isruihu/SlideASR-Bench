# ContextASR-Bench: A Massive Contextual Speech Recognition Benchmark  
<p align="center" dir="auto">
<a href="https://arxiv.org/abs/2510.08618" rel="nofollow"><img src="https://img.shields.io/badge/ArXiv-2507.05727-red" style="max-width: 100%;"></a>
<a href="https://huggingface.co/datasets/RUIH/SlideASR-Bench" rel="nofollow"><img src="https://img.shields.io/badge/Dataset-SlideASR_Bench-yellow" style="max-width: 100%;"></a>
<a href="https://huggingface.co/datasets/RUIH/VAPO-7B" rel="nofollow"><img src="https://img.shields.io/badge/Model-VAPO-blue" style="max-width: 100%;"></a>
</p>

## Download SlideASR-Bench Data  
The SildeASR-Bench dataset is now available for download at [🤗Huggingface🤗](https://huggingface.co/datasets/RUIH/SlideASR-Bench). 

## Model

The VAPO models are now available for download.

| Model | Download URL |
| --- | --- |
| VAPO-3B | [🤗Huggingface🤗](https://huggingface.co/RUIH/VAPO-3B) |
| VAPO-7B | [🤗Huggingface🤗](https://huggingface.co/RUIH/VAPO-7B) |

## Get Results on SlideASR-Bench

```shell
CUDA_VISIBLE_DEVICES=0,1 python run.py --model {MODEL_NAME} --dataset SlideASR-S --models-per-gpu {MODELS_PER_GPU}
```

## Evaluation
```shell
bash slide_asr/SlideASR-Bench/evaluation/evaluate.sh {MODEL_NAME}
```

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
