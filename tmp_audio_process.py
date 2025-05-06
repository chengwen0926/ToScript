# import librosa
# import numpy as np
# from pydub import AudioSegment

# def detect_silence(audio_path, min_silence_dur=1.0, threshold=-40):
#     y, sr = librosa.load(audio_path, sr=None)
#     intervals = librosa.effects.split(y, top_db=-threshold, frame_length=2048, hop_length=512)
#     silence_points = [(start/sr, end/sr) for start, end in intervals]
#     return [s for s in silence_points if (s[1]-s[0]) >= min_silence_dur]

# def hybrid_split(audio_path, max_duration=30, silence_params={}):
#     audio = AudioSegment.from_file(audio_path)
#     silence_segments = detect_silence(audio_path, **silence_params)
    
#     output = []
#     last_cut = 0
#     for start, end in silence_segments:
#         current_dur = (start - last_cut)
#         if current_dur > max_duration:
#             # 强制按最大时长分段
#             num_forced = int(current_dur // max_duration)
#             for i in range(num_forced):
#                 output.append(audio[last_cut*1000 : (last_cut + max_duration)*1000])
#                 last_cut += max_duration
#         # 添加静音分段点
#         output.append(audio[last_cut*1000 : start*1000]) 
#         last_cut = end
#     return output

# from pydub import AudioSegment
# import os

# def save_split_segments(segments, output_folder, file_prefix="segment", format="mp3"):
#     """
#     保存分割后的音频片段到指定文件夹
#     :param segments: hybrid_split返回的AudioSegment对象列表
#     :param output_folder: 目标文件夹路径
#     :param file_prefix: 文件名前缀（默认为"segment"）
#     :param format: 音频格式（支持mp3/wav等）
#     :return: 保存成功的文件路径列表
#     """
#     try:
#         # 创建目标文件夹（支持多级目录）
#         os.makedirs(output_folder, exist_ok=True)
        
#         saved_files = []
#         for i, segment in enumerate(segments, 1):
#             # 生成带序号的文件名（如segment_001.mp3）
#             filename = f"{file_prefix}_{i:03d}.{format}"
#             output_path = os.path.join(output_folder, filename)
            
#             # 导出音频并记录路径
#             segment.export(output_path, format=format)
#             saved_files.append(output_path)
        
#         return saved_files
#     except Exception as e:
#         print(f"保存失败：{str(e)}")
#         return []
    
# def hybrid_split_with_save(audio_path, output_folder, max_duration=30, silence_params={}, format="mp3"):
#     # 执行原有分割逻辑
#     segments = hybrid_split(audio_path, max_duration, silence_params)
    
#     # 保存分割结果
#     saved_files = save_split_segments(segments, output_folder, format=format)
    
#     print(f"成功保存 {len(saved_files)} 个音频片段到：{output_folder}")
#     return saved_files

# if __name__=='__main__':
#     saved_paths = hybrid_split_with_save(
#     audio_path="asset/audio.mp3",
#     output_folder="./output_segments",
#     max_duration=30,
#     silence_params={"threshold": -40, "min_silence_dur": 0.5},
#     format="wav"
# )

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
from pydub import AudioSegment
import os


def hybrid_split_with_duration_limit(segments, max_duration=30):
    """时长限制切割（新增核心函数）"""
    processed_segments = []
    for seg in segments:
        duration_ms = len(seg)
        if duration_ms <= max_duration*1000:  # 合格片段直接保留
            processed_segments.append(seg)
        else:  # 超长片段强制切割
            chunk_size = max_duration * 1000
            num_chunks = duration_ms // chunk_size + 1
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i+1)*chunk_size, duration_ms)
                processed_segments.append(seg[start:end])
    return processed_segments


def split_audio_preserve_samplerate(
    input_path: str, 
    start_time: float = 0.0,
    end_time: float = None
) -> str:
    """增强版支持时间区间筛选的音频切分函数[1,4,7](@ref)
    
    参数新增：
        start_time (float): 处理起始时间（秒）
        end_time (float): 处理结束时间（秒，None表示文件末尾）
    """
    try:
        # 输入验证增强（网页7）
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"文件不存在: {input_path}")
        
        # 加载音频并截取时间区间（关键修改点）
        audio = AudioSegment.from_file(input_path)
        original_sr = audio.frame_rate
        
        # 时间单位转换（秒→毫秒）[4](@ref)
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000) if end_time else len(audio)
        
        # 区间有效性校验
        if start_ms < 0 or start_ms >= len(audio):
            raise ValueError("起始时间超出音频范围")
        if end_ms > len(audio):
            end_ms = len(audio)
        
        # 截取目标区间音频（网页4核心逻辑）
        target_audio = audio[start_ms:end_ms]

        # 创建输出目录（保持原逻辑）
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(os.path.dirname(input_path), 
                                 f"{base_name}_{start_time}-{end_time}_segments")
        os.makedirs(output_dir, exist_ok=True)

        # 区间内静音检测（网页5/网页7逻辑增强）
        speech_ranges = detect_nonsilent(
            target_audio,  # 关键修改：使用截取后的音频
            min_silence_len=1000,
            silence_thresh=-40,
            seek_step=10
        )

        # 生成区间内切分点（新增时间偏移补偿）
        split_points = []
        prev_end = 0
        for i, (start, end) in enumerate(speech_ranges):
            # 计算全局时间戳（相对于原始音频）
            global_start = start_ms + start
            global_end = start_ms + end
            
            if i > 0 and (start - prev_end) >= 1000:
                split_start = prev_end + 500 + start_ms  # 偏移补偿
                split_end = start - 500 + start_ms
                split_points.append((split_start, split_end))
            prev_end = end

        # 执行区间切分（网页2/网页7增强逻辑）
        segments = []
        last_cut = start_ms  # 关键修改：起始点设为用户指定时间
        for cut_start, cut_end in split_points:
            if cut_start > end_ms:  # 超出处理区间则终止
                break
            segment = audio[last_cut:min(cut_end, end_ms)]  # 区间边界保护
            segments.append(segment)
            last_cut = cut_end
        
        # 添加最后一段（区间尾部处理）
        if last_cut < end_ms:
            segments.append(audio[last_cut:end_ms])

        segments = hybrid_split_with_duration_limit(segments, max_duration=30)

        # 导出处理（保持采样率核心逻辑）[7](@ref)
        for idx, seg in enumerate(segments):
            output_path = os.path.join(output_dir, 
                f"seg_{start_time}-{end_time}_{idx:03d}{os.path.splitext(input_path)[1]}")
            seg.export(output_path, 
                      format=os.path.splitext(input_path)[1][1:],
                      parameters=["-ar", str(original_sr)])

        return output_dir

    except Exception as e:
        print(f"处理失败: {str(e)}")
        return ""

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os

def voice_segmentor(input_path: str, output_dir: str, start_time: float = 0.0, end_time: float = None) -> list:
    """智能人声保留切割器"""
    try:
        # 音频加载与参数初始化
        audio = AudioSegment.from_file(input_path)
        original_sr = audio.frame_rate

        # 时间单位转换（秒→毫秒）[4](@ref)
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000) if end_time else len(audio)

        # 区间有效性校验
        if start_ms < 0 or start_ms >= len(audio):
            raise ValueError("起始时间超出音频范围")
        if end_ms > len(audio):
            end_ms = len(audio)
        
        # 截取目标区间音频（网页4核心逻辑）
        audio = audio[start_ms:end_ms]
        
        # 增强型静音检测（网页3/7）
        silence_params = {
            'min_silence_len': 800,   # 静音段最小800ms
            'silence_thresh': -35,    # 阈值降低至-35dB
            'seek_step': 15          # 检测步长缩短到15ms
        }
        
        # 人声区间检测（网页4核心逻辑）
        voice_ranges = detect_nonsilent(
            audio,
            **silence_params
        )
        
        # 生成有效人声片段
        segments = []
        for start_ms, end_ms in voice_ranges:
            # 添加500ms缓冲（网页7建议）
            safe_start = max(0, start_ms - 500)
            safe_end = min(len(audio), end_ms + 500)
            segments.append(audio[safe_start:safe_end])
            
        # 强制时长限制（网页1补充）
        MAX_DURATION = 30 * 1000  # 30秒限制
        final_segments = []
        for seg in segments:
            if len(seg) > MAX_DURATION:
                # 按30秒强制切割（网页3方法）
                for chunk in seg[::MAX_DURATION]:
                    final_segments.append(chunk)
            else:
                final_segments.append(seg)
                
        # 导出处理
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        for idx, seg in enumerate(final_segments):
            output_path = os.path.join(output_dir, 
                f"voice_seg_{idx:03d}{os.path.splitext(input_path)[1]}")
            seg.export(output_path, 
                format=os.path.splitext(input_path)[1][1:],
                parameters=["-ar", str(original_sr)])
            saved_files.append(output_path)
            
        return saved_files
    
    except Exception as e:
        print(f"处理失败: {str(e)}")
        return []
    
if __name__ == '__main__':
    audio_path = 'asset/audio.mp3'
    # res = split_audio_preserve_samplerate(audio_path)
    # res = split_audio_preserve_samplerate(audio_path,51)
    # print(res)
    voice_segmentor(audio_path, './outputs', 50)