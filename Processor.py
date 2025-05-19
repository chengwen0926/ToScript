from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import librosa
import numpy as np
import config
import logging
import utils
import os
from pathlib import Path
import ffmpeg
import subprocess


logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class AudioProcessor:
    def __init__(self, audio_file, backend='pydub'):
        self.backend = backend
        self.audio_file = audio_file

        if self.backend == 'pydub':
            self.audio = AudioSegment.from_file(audio_file)
            self.sample_rate = self.audio.frame_rate
            self.duration = self.audio.duration_seconds
            self.extension = os.path.splitext(self.audio_file)[1]
            
            if self.extension not in config.AUDIO_EXTENSIONS:
                raise ValueError(f"[Enlight] AudioProcessor类尚不支持处理{config.AUDIO_EXTENSIONS}以外的音频类型")
        else:
            raise ValueError(f"[Enlight] AudioProcessor类实现尚不支持pydub以外的库")

    def slice_audio(
        self,
        save:bool = False,
        local_dir: str = '', 
        maximum_duration:float = 30, 
        slice_start: float = 0.0, 
        slice_end: float = None,
        buffer: int = 250
    ):
        '''
            对初始化上传的音频文件进行片段切分操作
        Args:
            save: 是否将切分后的音频片段进行保存，如果是的话需要填写参数local_dir
            local_dir: 音频文件保存的本地目录
            maximum_duration: 音频片段的最大时长，单位为秒s
            slice_start: 开始执行切分操作的时间位置，单位为秒s
            slice_end: 结束执行切分操作的时间位置，单位为秒s
            buffer: 切分片段两边保留的缓冲时间，单位为毫秒ms


        Returns:
            segments: 切分后的音频列表
        '''
        logging.info(f'开始切分音频文件 {self.audio_file}')
        try:
            slice_start_ms = int(slice_start * 1000)
            slice_end_ms = int(slice_end * 1000) if slice_end else len(self.audio)
            maximum_duration_ms = int(maximum_duration * 1000)

            if slice_start_ms<0 or slice_start_ms>=len(self.audio):
                raise ValueError(f"[Enlight] slice_start超出时间原有音频范围")
            if slice_end_ms>len(self.audio):
                slice_end_ms = len(self.audio)
            elif slice_end_ms<0:
                slice_end_ms = len(self.audio) + slice_end_ms

            slice_audio = self.audio[slice_start_ms: slice_end_ms]
            silence_params = {
                'min_silence_len': 800,   # 静音段最小800ms
                'silence_thresh': -35,    # 阈值降低至-35dB
                'seek_step': 15          # 检测步长缩短到15ms
            }
            voice_ranges = detect_nonsilent(
                slice_audio,
                **silence_params
            )
            segments = []
            for seg_start_ms, seg_end_ms in voice_ranges:
                # 添加缓冲时间
                safe_start = max(0, seg_start_ms - buffer)
                safe_end = min(len(slice_audio), seg_end_ms + buffer)
                segments.append(slice_audio[safe_start:safe_end])
            
            # 再一次切分，将超过maximum_duration的片段进行切分
            final_segments = []
            for segment in segments:
                if len(segment) > maximum_duration_ms:
                    for chunk in segment[::maximum_duration_ms]:
                        final_segments.append(chunk)
                else:
                    final_segments.append(segment)
            
            if save:
                utils.check_path(local_dir)
                saved_files = []
                for idx, seg in enumerate(final_segments):
                    output_path = os.path.join(local_dir,  f"audio_seg_{idx:04d}{self.extension}")
                    seg.export(output_path, 
                        format=self.extension[1:],
                        parameters=["-ar", str(self.sample_rate)])
                    saved_files.append(output_path)
            segments = final_segments
            return segments

        except Exception as e:
            logging.error(f'[Enlight] 音频切分失败 \n {str(e)}')
            return None

    @classmethod
    def detect_audio_files(
        cls, 
        path:str, 
        extension:list[str]=config.AUDIO_EXTENSIONS, 
        recursive: bool = True
    )->list:
        '''
            类函数：读取指定目录下所有符合类型要求的文件路径，支持递归读取
        
        Args:
            path: 读取路径
            extension: 待读取的文件类型
            recursive: 递归读取

        Returns:
            audio_files: 读取到的包含有所有符合文件的路径列表
        '''
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"[Enlight] {path} 路径不存在")
            if not os.path.isdir(path):
                raise NotADirectoryError(f"[Enlight] {path} 路径不是文件夹目录")
            
            audio_extensions = extension
            audio_files = []
            search_method = Path(path).rglob('*') if recursive else Path(path).glob('*')
            
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
            logging.error(f'[Enlight] 检测音频文件时出现报错\n {str(e)}')
            return []
        
    @classmethod
    def check_audio_quality(cls, audio_file:str):
        '''
            输出音频参数

        Args:
            audio_file: 音频文件路径
        '''
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_name,sample_rate,channels,bits_per_sample',
            '-of', 'default=noprint_wrappers=1',
            audio_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)


class VideoProcessor:
    def __init__(self, video_file):
        self.video_file = video_file

    def separate_audio(
        self,
        local_dir: str = ''
    ):
        '''
            从视频文件中分离音频并保存
        Args:
            local_dir: 音频文件保存的本地目录

        Returns:

        '''
        probe = ffmpeg.probe(self.video_file)
        audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

        audio_file = 'separate_audio.wav'
        utils.check_path(local_dir)
        output_audio = os.path.join(local_dir, audio_file)

        if audio_stream and audio_stream['codec_name'] in ['pcm_s16le', 'flac']:
            # 直接复制原始无损音频流
            (ffmpeg.input(self.video_file)
            .output(output_audio, acodec='copy', loglevel="error")
            .run(overwrite_output=True))
        else:
            # 转码为无损WAV格式
            (ffmpeg.input(self.video_file)
            .output(output_audio, 
                    acodec='pcm_s16le',  # 16位无损PCM编码
                    ar='44100',          # 采样率44.1kHz（CD标准）
                    ac=2,               # 立体声
                    loglevel="error")
            .run(overwrite_output=True))


if __name__=='__main__':
    # 示例调用

    # 视频音频分离
    # vp = VideoProcessor('./asset/哪吒之魔童降世.mp4')
    # vp.separate_audio('./outputs')

    # 音频参数展示
    # AudioProcessor.check_audio_quality('./outputs/separate_audio.wav')

    # 音频内容切分
    audio_file = './outputs/separate_audio.wav'
    AP = AudioProcessor('./outputs/separate_audio.wav')
    AP.slice_audio(
        save=True,
        local_dir='./outputs/nezha_audio_slice',
        maximum_duration=29,
        slice_start=95,
        slice_end=-400,
        buffer=250
    )