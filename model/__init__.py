import transformers
VERSION = transformers.__version__

from .qwen2_5_omni import (
    Qwen2_5OmniWorker,
    Qwen2_5OmniAVWorker,
    VAPOWorker,
)
from .qwen2_audio import Qwen2AudioWorker, Qwen2AudioWithOCRWorker
from .midashenglm import MiDashengWorker, MiDashengWithOCRWorker


if VERSION >= '4.54.0':
    from .qwen3_omni import Qwen3OmniWorker, Qwen3OmniAVWorker, Qwen3OmniWithOCRWorker
else:
    Qwen3OmniWorker, Qwen3OmniAVWorker, Qwen3OmniWithOCRWorker = None, None, None
