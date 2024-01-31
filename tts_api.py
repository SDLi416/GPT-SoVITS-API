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

from tts_inference import start_infer

class TextToSpeechRequest(BaseModel):
    speaker: str = Field('yuze', description="说话者")
    text: str = Field(..., description="需要转换的文字")
    language: str = Field('Chinese', description="语言")
    # speed: float = Field(1.0, description="语速")
    # model_name: str = "tts_model"  # 设置默认模型名称

app = FastAPI()

def process_audio_in_memory(audio_data, sampling_rate):
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

@app.post("/tts")
async def text_to_speech(request: TextToSpeechRequest):
    # 运行模型推理
    try:
        sampling_rate, audio_data = start_infer(request.speaker, request.text, request.language)

        # wav = io.BytesIO()

        # sf.write(wav, audio_data, sampling_rate, format="wav")

        # wav.seek(0)

        # return StreamingResponse(wav, media_type="audio/wav")
        # 将模型输出转换为适当的音频格式
        # 示例中假设输出是音频数据的 numpy 数组
        stream_buffer, duration = process_audio_in_memory(audio_data, sampling_rate)
        response = StreamingResponse(stream_buffer, media_type="audio/mpeg")
        response.headers["Duration"] = str(duration)
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)