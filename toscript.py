
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
import utils
from model import Model
import processor
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

CATEGORY = [
        'Audio2Script',
        'Image2Script',
        'Video2Script'
    ]
DEFAULT_MODEL = {
    'Audio2Script':'whisper-large-v3-turbo',
    'Image2Script':'',
    'Video2Script':'whisper-large-v3-turbo'
}


class ToScript:

    def __init__(
        self, 
        source:str, 
        repo_id:str=None,
        save:bool=True,
        local_dir:str='./'
    ):
        '''

        Args:
            source: 需要被处理成文本的文件
            repo_id: 模型的Repository ID
            save: 转写得到的文本是否保存，如果是的话需要填写参数local_dir
            local_dir: 文本文件保存的本地目录
        Returns:
            
        '''
        self.source = source
        self.filename = os.path.splitext(self.source)[0]
        self.extension = os.path.splitext(self.source)[1]
        if self.extension in config.AUDIO_EXTENSIONS: # TODO 补充完善音频、图像、视频的处理逻辑
            self.category = 'Audio2Script'
        elif self.extension in config.AUDIO_EXTENSIONS:
            self.category = 'Image2Script'
        elif self.extension in config.VIDEO_EXTENSIONS:
            self.category = 'Video2Script'
        self.repo_id = repo_id if repo_id else DEFAULT_MODEL[self.category]
        self.model = Model(self.repo_id)
        self.save = save
        self.local_dir = local_dir

    
    def execute(self)->list[str]:
        '''
            执行 to script 操作
        Returns:
            res: 转写得到的文本组成的列表
        '''
        # 获取时间戳
        timestamp = time.time()
        struct_time = time.localtime(timestamp)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", struct_time)

        logging.info('''[Enlight] 开始执行转文本操作，具体任务信息如下\n
                     开始时间 - {timestamp}\n
                     处理文件 - {self.source}\n
                     任务类型 - {self.category}\n
                     使用模型 - {self.repo_id}\n
                     ''')
        tmp_output_dir = './outputs/tmp/'.format(self.filename+' '+timestamp)
        output_dir = './outputs/{}'.format(self.filename+' '+timestamp)
        output_file = os.path.join(self.local_dir, 'result.txt')

        if self.category == 'Audio2Script':
            ap = processor.AudioProcessor(self.source)
            ap.slice_audio(
                save=True,
                local_dir=output_dir, # TODO 添加对开头和结尾的手动标记
            )
            files = processor.AudioProcessor.detect_audio_files(output_dir)
            res = []
            for file in files:
                content = self.model.invoke(file)
                res.append(content)
                if self.save:
                    utils.append_str_to_file(content, output_file)
            
            logging.info(f'[Enlight] {self.category}执行完毕')
            return res
            
        elif self.category == 'Image2Script':
            logging.error(f'[Enlight] {self.category}逻辑尚未完成') # TODO 完善相关逻辑

        elif self.category == 'Video2Script':
            vp = processor.VideoProcessor(self.source)
            separate_audio_dir = vp.separate_audio(tmp_output_dir)
            ap = processor.AudioProcessor(separate_audio_dir)
            ap.slice_audio(
                save=True,
                local_dir=output_dir, # TODO 添加对开头和结尾的手动标记
            )
            files = processor.AudioProcessor.detect_audio_files(output_dir)
            res = []
            for file in files:
                content = self.model.invoke(file)
                res.append(content)
                if self.save:
                    utils.append_str_to_file(content, output_file)

            logging.info(f'[Enlight] {self.category}执行完毕')
            return res


if __name__=='__main__':
    pass