import os
import config
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import logging


# 模型相关操作
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def download_repo(
        repo_id:str, 
        local_dir:str=config.PRETRAINED_MODEL_ROOT_DIR, 
        max_retries:int=3, 
        hf_token:str=''
    ):
    '''
    根据repo_id下载HuggingFace中的模型仓库

    Args:
        repo_id: huggingface中的存储库ID
        local_dir: 模型下载到本地的路径
        max_retries: 网络问题出现下载失败时的最大重试次数
        hf_token: 个人huggingface token

    Returns:
        local_dir: 模型下载成功后存放在本地的目录 若下载失败则返回""
    '''
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    check_path(local_dir)
    local_dir = os.path.join(local_dir, repo_id.split('/')[-1])

    # 参考 https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.snapshot_download
    for attempt in range(1,max_retries+1):
        logging.info(f"[Enlight] 开始下载: {repo_id} (尝试次数: {attempt}/{max_retries})")
        try:
            if hf_token:
                huggingface_hub.snapshot_download(
                    repo_id=repo_id, 
                    local_dir=local_dir,
                    resume_download=True, # 启用断点续传
                )
            else:
                huggingface_hub.snapshot_download(
                    repo_id=repo_id, 
                    local_dir=local_dir,
                    resume_download=True, # 启用断点续传
                    token=hf_token
                )
            logging.info(f"[Enlight] 下载成功: {repo_id}")
            return local_dir
        except Exception as e:
            logging.error(f"[Enlight] 下载失败: {str(e)}")
            if attempt<=max_retries:
                retry_delay = 10 * (attempt + 1)
                logging.info(f"[Enlight] {retry_delay}秒后重试...")
                time.sleep(retry_delay)
            else:
                logging.error(f"[Enlight] 下载失败, 已达到最大重试次数，可尝试手动将 {repo_id} 下载到 {local_dir}")
                return ""
            

class Model:
    '''
        用于封装'Audio2Script'、'Image2Script'、'Video2Script'三类模型的类
    '''
    def __init__(self, repo_id:str):
        self.model_dir = os.path.join(config.PRETRAINED_MODEL_ROOT_DIR, repo_id)
        self.language = 'chinese'
        if not os.path.exists(self.model_dir): # TODO 只存在对应目录还不够，应该添加检查模型文件是否完整的逻辑
            download_repo(repo_id=repo_id)

    def invoke(self, source:str):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # 加载模型
        try:
            # TODO 确定是不是所有音频转文本的模型都可以通过AutoModelForSpeechSeq2Seq来调用
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(device)
            processor = AutoProcessor.from_pretrained(self.model_dir)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
            )
            result = pipe(source, generate_kwargs={"language": self.language})
            return result
        except Exception as e:
            logging.error(f'[Enlight] 模型调用报错 \n {str(e)}')
