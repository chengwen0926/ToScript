# from modelscope import snapshot_download
# import os

# class ToScript:

#     def __init__(self, model_dir):
#         self.instruct = True if '-Instruct' in model_dir else False
#         self.model_dir = model_dir
#         # self.fp16 = fp16
#         if not os.path.exists(model_dir):
#             model_dir = snapshot_download(model_dir)
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import huggingface_hub
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import config
from pydub import AudioSegment


def download_repo(repo_id):
    local_dir = os.path.join(config.PRETRAINED_MODEL_ROOT_DIR, repo_id.split('/')[-1])
    # 参考 https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.snapshot_download
    huggingface_hub.snapshot_download(repo_id, local_dir=local_dir)
    return local_dir


def check_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000  # 毫秒转秒
    return duration_seconds < config.MAX_AUDIO_DURATION


def audio2txt(audio_file, model): # 可以是本地模型也可以是repo_id
        # 检查音频类型和时长
        if audio_file.is_file() and audio_file.suffix.lower() in config.AUDIO_EXTENSIONS and check_audio_duration(audio_file):
            pass
        else:
            raise Exception('[Enlight] 音频不符合要求')
        
        # 检查模型是在本地运行还是huggingface上的仓库
        model_dir = ''
        if os.path.exists(model) or os.path.exists(os.path.join(config.PRETRAINED_MODEL_ROOT_DIR, model)):
            model_dir = model
        else:
            try:
                model_dir = download_repo(model)
            except Exception as e:
                print(f'[Enlight] 模型下载失败 \n {e}')
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # 加载模型
        # TODO 确定是不是所有音频转文本的模型都可以通过AutoModelForSpeechSeq2Seq来调用
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(device)
            processor = AutoProcessor.from_pretrained(model_dir)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
            )
            result = pipe(audio_file)
            return result
        except Exception as e:
            print(f'[Enlight] 模型调用报错 \n {e}')


class AudioProcesser:
    '''
    源音频文件所在的目录
    音频划分策略（最长单个片段时间、划分逻辑）
        Strategy 类来记录相应的划分策略
    目标文件夹位置
        命名格式
        存储类型
    选择的模型
        whisper
        ...

    '''
    def __init__(self):
        pass


    



if __name__=='__main__':
    download("pretrained_models/whisper-large-v3-turbo")