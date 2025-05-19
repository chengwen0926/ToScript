from pydub import AudioSegment
import librosa
import numpy as np


class AudioProcessor:
    def __init__(self, backend='pydub'):
        self.audio_data = None
        self.sample_rate = 44100
        self.backend = backend
    
    def load_audio(self, file_path: str) -> np.ndarray:
        '''
        将音频文件加载得到对应的numpy数据格式的文件

        '''
        if self.backend == 'pydub':
            audio = AudioSegment.from_file(file_path)
            self.audio_data = np.array(audio.get_array_of_samples())
        elif self.backend == 'librosa':
            self.audio_data, self.sample_rate = librosa.load(file_path, sr=None)
        return self.audio_data
'''
AudioProcessor
    获取传入音频的信息
        check_audio_duration
    对音频进行切分
    获取指定文件夹中所有音乐类型的内容

Model
'''