import os,sys
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS"%(now_dir))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import io
import soundfile as sf
from pydub import AudioSegment
import numpy as np

from tts_inference import start_infer, start_few_shot_infer

class TextToSpeechRequest(BaseModel):
    speaker: str = Field('yuze', description="说话者")
    text: str = Field(..., description="需要转换的文字")
    fewshot: bool = Field(False, description="是否使用few shot模式")
    language: str = Field('Chinese', description="语言")
    # speed: float = Field(1.0, description="语速")
    # model_name: str = "tts_model"  # 设置默认模型名称

app = FastAPI()

# 检查音频是否有效
def is_audio_valid(audio_data, sampling_rate):
    # 检查音频数据是否非空
    if audio_data is None or len(audio_data) == 0:
        return False

    # 检查音频是否几乎完全静音
    # 这里假设音频数据是单通道；如果是多通道，需要相应调整
    if np.max(np.abs(audio_data)) < 0.01:  # 阈值可根据实际情况调整
        return False

    # 尝试解码音频数据（可选）
    try:
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sampling_rate, format='wav')
        buffer.seek(0)
        # 如果无法读取音频数据，soundfile会抛出错误
        _, _ = sf.read(buffer)
    except Exception as e:
        return False

    # 如果所有检查都通过，则认为音频有效
    return True

def process_audio_in_memory(audio_data, sampling_rate):
    if not is_audio_valid(audio_data, sampling_rate):
        return None, 0  # 音频无效，直接返回None和0时长

    # 将音频数据写入buffer
    buffer = io.BytesIO()

    # 这里必须使用soundfile来处理，否则会有噪音
    sf.write(buffer, audio_data, sampling_rate, format='wav')    
    # 重置缓冲区的读取位置
    buffer.seek(0)

    # 读取缓冲区中的音频数据
    audio_segment = AudioSegment.from_wav(buffer)

    # 将音频转换为 MP3 并保存到新的 io.BytesIO 对象
    mp3_buffer = io.BytesIO()
    audio_segment.export(mp3_buffer, format="mp3")
    mp3_buffer.seek(0)

    # 计算音频时长
    duration = len(audio_segment) / 1000

    return (mp3_buffer, duration)

# 加入重试机制的生成音频函数
async def generate_audio(speaker, text, language, few_shot=False, max_retries=3):
    for _ in range(max_retries):
        try:
            if few_shot:
                sampling_rate, audio_data = start_few_shot_infer(speaker, text, language)
            else:
                sampling_rate, audio_data = start_infer(speaker, text, language)
            stream_buffer, duration = process_audio_in_memory(audio_data, sampling_rate)
            if stream_buffer:
                return stream_buffer, duration
        except Exception as e:
            print(f"Error generating audio: {e}")
    return None, 0  # 所有尝试都失败后返回None

@app.post("/tts")
async def text_to_speech(request: TextToSpeechRequest):
    stream_buffer, duration = await generate_audio(request.speaker, request.text, request.language, request.fewshot)
    if stream_buffer:
        response = StreamingResponse(stream_buffer, media_type="audio/mpeg")
        response.headers["Duration"] = str(duration)
        return response
    else:
        raise HTTPException(status_code=500, detail="Failed to generate valid audio after multiple attempts")

@app.post("/tts-ext")
async def text_to_speech_extended(request: TextToSpeechRequest):
    stream_buffer, duration = await generate_audio(request.speaker, request.text, request.language, few_shot=True)
    if stream_buffer:
        response = StreamingResponse(stream_buffer, media_type="audio/mpeg")
        response.headers["Duration"] = str(duration)
        return response
    else:
        raise HTTPException(status_code=500, detail="Failed to generate valid audio after multiple attempts")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)