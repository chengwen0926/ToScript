'''
调用本地whisper模型
'''
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
# 报错 moviepy.editor 无法载入
# 新版本中取消了editor 安装此前版本的moviepy库 pip install moviepy==1.0.3
from moviepy.editor import VideoFileClip
from typing import Optional
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.utils import make_chunks  # 新增模块导入[4,5](@ref)
import traceback
from pathlib import Path
import logging

def append_to_file(file_path: str, content: str) -> bool:
    """
    将指定内容追加写入文本文件（自动创建不存在的文件/目录）
    
    参数：
    - file_path : 目标文件完整路径（如：/data/logs/app.log）
    - content   : 要追加的文本内容
    
    返回值：
    - 写入成功返回True，失败返回False
    
    功能特性：
    - 自动创建不存在的父目录[6,7](@ref)
    - 自动处理中文路径编码[4](@ref)
    - 捕获权限不足/磁盘已满等异常[9,10](@ref)
    - 支持任意可转换为字符串的数据类型[3,5](@ref)
    """
    try:
        # 路径标准化处理
        target_file = Path(file_path)
        
        # 自动创建父目录（支持多级目录创建）
        target_file.parent.mkdir(parents=True, exist_ok=True)  # [6,8](@ref)
        
        # 内容预处理（支持非字符串类型）
        content_str = f"\n{str(content).strip()}"  # 自动添加换行符
        
        # 执行追加写入（显式指定UTF-8编码）
        with target_file.open("a", encoding="utf-8") as f:  # [2,4](@ref)
            f.write(content_str)
            
        return True
        
    except PermissionError as e:
        logging.error(f"权限不足: {file_path} - {str(e)}")  # [9](@ref)
        return False
    except IOError as e:
        logging.error(f"I/O错误: {file_path} - {str(e)}")   # [9,10](@ref)
        return False
    except Exception as e:
        logging.error(f"未知错误: {file_path} - {str(e)}")   # [9](@ref)
        return False
    
def extract_audio(video_path: str) -> Optional[str]:
    """
    从视频文件中分离音频并保存为MP3格式
    :param video_path: 视频文件路径(支持mp4/avi/mov/mkv等常见格式)
    :return: 成功返回音频文件路径，失败返回None
    """
    try:
        # 异常1：文件不存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 异常2：路径指向目录而非文件
        if os.path.isdir(video_path):
            raise IsADirectoryError(f"路径指向的是目录: {video_path}")

        # 加载视频文件（自动处理格式兼容性问题）
        with VideoFileClip(video_path) as video:
            # 异常3：视频不含音频轨道
            if video.audio is None:
                raise ValueError("视频文件没有音频轨道")

            # 生成输出路径（保留原文件名）
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(
                os.path.dirname(video_path),
                f"{base_name}_audio.mp3"
            )

            # 写入音频文件（自动处理编解码器）
            video.audio.write_audiofile(output_path)
            
            return output_path

    except (FileNotFoundError, IsADirectoryError) as e:
        print(f"路径错误: {str(e)}")
    except IOError as e:
        print(f"文件读写错误: {str(e)}")
    except ValueError as e:
        print(f"内容错误: {str(e)}")
    except Exception as e:
        print(f"未知错误: {str(e)}")
        # 可添加日志记录模块
        # logging.error(f"音频分离失败: {str(e)}", exc_info=True)
    
    return None

def split_audio_preserve_samplerate(input_path: str) -> str:
    """保持原始采样率的音频切分函数"""
    try:
        # 输入验证
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"文件不存在: {input_path}")
        
        # 加载音频并获取原始采样率（网页4/网页7）
        audio = AudioSegment.from_file(input_path)
        original_sr = audio.frame_rate  # 关键修改点：记录原始采样率
        
        # 创建输出目录
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(os.path.dirname(input_path), f"{base_name}_segments")
        os.makedirs(output_dir, exist_ok=True)

        # 分割逻辑（网页5/网页7）
        speech_ranges = detect_nonsilent(
            audio,
            min_silence_len=1000,  # 静音间隙阈值1秒
            silence_thresh=-40,     # 分贝阈值
            seek_step=10            # 检测步长10ms
        )
        
        # 生成切分点
        split_points = []
        prev_end = 0
        # for start, end in speech_ranges:
        #     if start - prev_end >= 1000:  # 有效语音间隙
        #         split_points.append((prev_end, start))
        #     prev_end = end
        for i, (start, end) in enumerate(speech_ranges):
            if i > 0 and (start - prev_end) >= 1000:  # 检测到有效语音间隙
                split_points.append((prev_end + 500, start - 500))  # 保留500ms缓冲
            prev_end = end

        # 执行切分并保持采样率（网页2/网页4）
        segments = []
        last_cut = 0
        for cut_start, cut_end in split_points:
            segment = audio[last_cut:cut_end]
            segments.append(segment)
            last_cut = cut_end
        segments.append(audio[last_cut:])  # 添加最后一段

        # 导出处理（关键修改点：显式设置采样率）
        for idx, seg in enumerate(segments):
            output_path = os.path.join(output_dir, f"seg_{idx:03d}{os.path.splitext(input_path)[1]}")
            seg.export(output_path, 
                      format=os.path.splitext(input_path)[1][1:],
                      parameters=["-ar", str(original_sr)])  # 强制指定采样率

        return output_dir

    except Exception as e:
        print(f"处理失败: {str(e)}")
        return ""
   

def get_audio_files(folder_path: str, recursive: bool = True) -> list:
    """
    获取指定文件夹中的所有音频文件路径（支持递归搜索）
    
    参数：
    - folder_path : 目标文件夹路径（字符串）
    - recursive : 是否递归子目录（默认启用）
    
    返回值：
    - 包含有效音频文件绝对路径的列表（空列表表示无结果或异常）
    
    支持格式：MP3/WAV/FLAC/AAC/OGG/WMA
    """
    try:
        # 异常预检（网页8）
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"目录不存在: {folder_path}")
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"路径不是目录: {folder_path}")

        # 定义支持的音频格式（网页1/2/5）
        audio_extensions = ('.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma')
        
        # 路径遍历核心逻辑（网页1/3/5）
        audio_files = []
        search_method = Path(folder_path).rglob('*') if recursive else Path(folder_path).glob('*')
        
        for file_path in search_method:
            try:
                # 格式验证与权限检查（网页8/10）
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    audio_files.append(str(file_path.resolve()))
            except PermissionError as e:
                print(f"权限拒绝访问: {file_path} ({str(e)})")
            except Exception as e:
                print(f"未知错误: {file_path} ({str(e)})")
        
        return audio_files

    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"路径错误: {str(e)}")
        return []
    except Exception as e:
        print(f"系统级错误: {str(e)}")
        return []

def generate_script(model_dir:str="pretrained_models/whisper-large-v3-turbo", audio_dir:str="asset/test_audio.wav"):
    # TODO 完善异常捕获相关的逻辑
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # model_id = "openai/whisper-large-v3-turbo"
    # model_dir = "pretrained_models/whisper-large-v3-turbo"

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
    result = pipe(audio_dir)
    return result["text"]

def audio_func_test(audio_files_dir:str):
    import time
    print('【开始转录音频文件】')
    s_time = time.time()
    files = get_audio_files(audio_files_dir, True)
    files.sort(reverse=False)
    for index, audio in enumerate(files):
        try:
            print('\t\t处理音频:{}'.format(audio))
            start = time.time()
            res = generate_script(audio_dir=audio)
            append_to_file('asset/record.txt',res)
            end = time.time()
            print('\t\t处理时间{},文本内容{}'.format(start-end, res))
        except Exception as e:
            print('处理失败{}'.format(audio))
            print(e)
    e_time = time.time()
    print('\t转录完成 花费时间{}'.format(e_time-s_time))


def full_process_test(video_path):
    import time
    init = time.time()

    print('【开始提取完整音频文件】')
    start = time.time()
    audio_file = extract_audio(video_path=video_path)
    end = time.time()
    print('\t提取完成 花费时间{}'.format(end-start))
    
    print('【开始分割音频文件】')
    start = time.time()
    audio_files_dir = split_audio_preserve_samplerate(audio_file)
    end = time.time()
    print('\t分割完成 花费时间{}'.format(end-start))
    
    print('【开始转录音频文件】')
    s_time = time.time()
    files = get_audio_files(audio_files_dir, True)
    files.sort(reverse=False)
    for index, audio in enumerate(files):
        try:
            print('\t\t处理音频:{}'.format(audio))
            start = time.time()
            res = generate_script(audio_dir=audio)
            append_to_file('asset/record.txt',res)
            end = time.time()
            print('\t\t处理时间{},文本内容{}'.format(start-end, res))
        except Exception as e:
            print('处理失败{}'.format(audio))
    e_time = time.time()
    print('\t转录完成 花费时间{}'.format(e_time-s_time))
        


if __name__ == '__main__':
    # audio_path = 'asset/test_audio.wav'
    # # res = split_audio_preserve_samplerate(audio_path)
    # res = split_audio_preserve_samplerate(audio_path)
    # print(res)

    path = 'asset'
    res = get_audio_files(path,True)
    print(len(res))

    # append_to_file('asset/record.txt','初始化语句')

    # path = 'asset/从21世纪安全撤离.mp4'
    # full_process_test(path)
    # path = 'asset/从21世纪安全撤离_audio_segments'
    # audio_func_test(path)
