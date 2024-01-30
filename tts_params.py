from pathlib import Path
import json

"""
将训练出来的结果放到models文件夹下，然后在这里配置模型的参数
按照如下格式配置：

config.json:
{
    "prompt_text": "your_prompt_text",
    "prompt_language": "zh/en/ja"
}

sovits_weights.pth: 模型权重文件
gpt_weights.ckpt: 模型权重文件
ref.wav: 参考音频文件

然后放到models/{speaker}/文件夹下
完整的结构类似于：
models
├── speaker1
│   ├── config.json
│   ├── gpt_weights.ckpt
│   ├── ref.wav
│   └── sovits_weights.pth
└── speaker2
    ├── config.json
    ├── gpt_weights.ckpt
    ├── ref.wav
    └── sovits_weights.pth

后续会导出成onnx再加载
"""

class TTSModel:
    ref_wav_path: str # 参考音频路径
    sovits_weights_path: str # pth文件路径
    gpt_weights_path: str # ckpt文件路径
    prompt_text: str # 参考音频对应的中文
    prompt_language: str # zh/en/ja

# 获取TTS模型必要的参数
def get_tts_model_params(name: str) -> TTSModel:
    model_dir = Path("models")
    model = TTSModel()
    model.ref_wav_path = f"{model_dir}/{name}/ref.wav"
    model.sovits_weights_path = f"{model_dir}/{name}/sovits_weights.pth"
    model.gpt_weights_path = f"{model_dir}/{name}/gpt_weights.ckpt"
    with open(f"{model_dir}/{name}/config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
        model.prompt_text = config["prompt_text"]
        model.prompt_language = config["prompt_language"]
    return model
