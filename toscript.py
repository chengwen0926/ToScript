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
import config
from pydub import AudioSegment
from pathlib import Path
import logging
from pydub.silence import detect_nonsilent

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def download_repo(repo_id):
    local_dir = os.path.join(config.PRETRAINED_MODEL_ROOT_DIR, repo_id.split('/')[-1])
    # 参考 https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.snapshot_download
    huggingface_hub.snapshot_download(repo_id, local_dir=local_dir)
    return local_dir


def check_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000  # 毫秒转秒
    return duration_seconds < config.MAX_AUDIO_DURATION


def single_audio_to_txt(audio_file, model, language='chinese'): # 可以是本地模型也可以是repo_id
        # 检查音频类型和时长
        if os.path.splitext(audio_file)[1] in config.AUDIO_EXTENSIONS and check_audio_duration(audio_file):
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
                logging.error(f'[Enlight] 模型下载失败 \n {str(e)}')
        
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
            result = pipe(audio_file, generate_kwargs={"language": language})
            return result
        except Exception as e:
            logging.error(f'[Enlight] 模型调用报错 \n {str(e)}')


def append_content_to_file(content: str, file_path: str) -> bool:
    try:
        target_file = Path(file_path)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        content_str = f"\n{str(content).strip()}"  # 自动添加换行符
        with target_file.open("a", encoding="utf-8") as f:
            f.write(content_str)
            
        return True
        
    except PermissionError as e:
        logging.error(f"[Enlight] 权限不足\n {file_path} - {str(e)}")
        return False
    except IOError as e:
        logging.error(f"[Enlight] I/O错误\n {file_path} - {str(e)}")
        return False
    except Exception as e:
        logging.error(f"[Enlight] 未知错误\n {file_path} - {str(e)}")
        return False 
    

def split_audio(
        audio_file:str,
        output_dir: str, 
        max_duration:float = 30, 
        start_time: float = 0.0, 
        end_time: float = None,
        buffer_time: int = 250
    ):
    try:
        # 音频加载与参数初始化
        audio = AudioSegment.from_file(audio_file)
        original_sr = audio.frame_rate # sample_rate

        # 时间单位转换（秒→毫秒）
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000) if end_time else len(audio)
        max_duration_ms = int(max_duration * 1000)

        # 区间有效性校验
        if start_ms < 0 or start_ms >= len(audio):
            raise ValueError('[Enlight] 起始时间超出音频范围')
        if end_ms > len(audio):
            end_ms = len(audio)
        
        audio = audio[start_ms:end_ms]

        # 人声区间检测
        silence_params = {
            'min_silence_len': 800,   # 静音段最小800ms
            'silence_thresh': -35,    # 阈值降低至-35dB
            'seek_step': 15          # 检测步长缩短到15ms
        }
        voice_ranges = detect_nonsilent(
            audio,
            **silence_params
        )
        segments = []
        for start_ms, end_ms in voice_ranges:
            # 添加缓冲时间
            safe_start = max(0, start_ms - buffer_time)
            safe_end = min(len(audio), end_ms + buffer_time)
            segments.append(audio[safe_start:safe_end])

        final_segments = []
        for seg in segments:
            if len(seg) > max_duration_ms:
                # 按照规定的最长切片时间进行切分
                for chunk in seg[::max_duration_ms]:
                    final_segments.append(chunk)
            else:
                final_segments.append(seg)
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        for idx, seg in enumerate(final_segments):
            output_path = os.path.join(output_dir,  f"audio_seg_{idx:03d}{os.path.splitext(audio_file)[1]}")
            seg.export(output_path, 
                format=os.path.splitext(audio_file)[1][1:],
                parameters=["-ar", str(original_sr)])
            saved_files.append(output_path)
            
        return True
    
    except Exception as e:
        logging.error(f'[Enlight] 音频切分失败 \n {str(e)}')
        return False

def get_audio_files(folder_path: str, recursive: bool = True) -> list:
    """
    获取指定文件夹中的所有音频文件路径（支持递归搜索）    
    支持格式：MP3/WAV/FLAC/AAC/OGG/WMA
    """
    try:
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"[Enlight] 目录不存在 {folder_path}")
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"[Enlight] 路径不是目录 {folder_path}")
        
        audio_extensions = config.AUDIO_EXTENSIONS
        audio_files = []
        search_method = Path(folder_path).rglob('*') if recursive else Path(folder_path).glob('*')
        
        for file_path in search_method:
            try:
                # 格式验证与权限检查
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    audio_files.append(str(file_path.resolve()))
            except PermissionError as e:
                logging.error(f'[Enlight] 权限拒绝访问：{file_path}\n {str(e)}')
            except Exception as e:
                logging.error(f'[Enlight] 未知错误：{file_path}\n {str(e)}')
    
        return audio_files

    except (FileNotFoundError, NotADirectoryError) as e:
        logging.error(f'[Enlight] 路径错误\n {str(e)}')
        return []
    except Exception as e:
        logging.error(f'[Enlight] 系统级错误\n {str(e)}')
        return []

def process():
    audio_path = './asset/哪吒之魔童降世_audio.mp3'
    output_dir = './outputs_nezha'
    model_dir = './pretrained_models/whisper-large-v3-turbo'
    txt_dir = './temp_record_nezha.txt'
    split_audio(
        audio_file = audio_path,
        output_dir = output_dir, 
        max_duration = 29, 
        start_time = 96, 
        end_time = None
    )
    audios = get_audio_files(output_dir)
    for audio in audios:
        res = single_audio_to_txt(audio_file=audio, model=model_dir)
        append_content_to_file(content=res, file_path=txt_dir)



# class AudioProcesser:
#     '''
#     源音频文件所在的目录
#     音频划分策略（最长单个片段时间、划分逻辑）
#         Strategy 类来记录相应的划分策略
#     目标文件夹位置
#         命名格式
#         存储类型
#     选择的模型
#         whisper
#         ...
#     '''
#     def __init__(self):
#         pass


if __name__=='__main__':
    process()