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
from typing import *
import os
import json
import time

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
            
# 路径相关操作
def check_path(path:str):
    dir_path = Path(path)
    if not dir_path.exists():
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"[Enlight] 目录创建成功：{dir_path}")
        except PermissionError:
            logging.info(f"[Enlight] 权限不足，无法创建目录：{dir_path}")
        except Exception as e:
            logging.error(f"[Enlight] 目录创建失败: {str(e)}")
    else:
        logging.info(f"[Enlight] 目录已存在：{dir_path}")


# 文件处理相关操作
def str_to_bool(s: str) -> bool:
    '''
    将输入的字符串转换成布尔类型
    :param s:
    :return:
    '''
    return s.lower() == 'true'


def all_digits(s: str) -> bool:
    '''
    判断输入字符串是否只包括数字和符号
    :param s:
    :return:
    '''
    valid_characters = set("0123456789")
    return all(char in valid_characters for char in s)


def all_symbols_or_digits(s: str) -> bool:
    '''
    判断输入字符串是否只包括数字和符号
    :param s:
    :return:
    '''
    valid_characters = set("0123456789!@#$%^&*()-_=+[]{}|;:'\",.<>?/`~")
    return all(char in valid_characters for char in s)


def append_dicts_to_txt(list: List[dict], file_path: str) -> None:
    with open(file_path, "a", encoding="utf-8") as file:
        for data in list:
            file.write(str(data) + "\n")
    file.close()


def append_list_to_txt(dataset: List, file_path: str) -> None:
    if not (file_path.endswith(".txt")):
        raise FileNotFoundError("[Enlight] File Not a Txt File.")

    with open(file_path, "a", encoding="utf-8") as file:
        for data in dataset:
            file.write(str(data) + "\n")
    file.close()

def load_content_from_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def load_content_from_txt(file_path: str) -> str:
    '''
    读取txt文件中的不同章节 切分后返回
    :param file_path: txt文本路径
    :return:
    '''
    if not (file_path.endswith(".txt") and os.path.exists(file_path)):
        raise FileNotFoundError("[Enlight] File Not Found or Not a Txt File.")

    raw_content = open(file_path, encoding="utf-8").read()
    lines = raw_content.split("\n")

    strip_content = ""
    for line in lines:
        strip_content = strip_content + line.strip() + "\n"

    return strip_content


def load_dicts_from_txt(file_path: str) -> List[dict]:
    '''
    从txt文件中提取元素为自字典类型的列表
    :param file_path:
    :return:
    '''
    extractions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        # 逐行读取
        for line in file:
            data = line.strip().replace('\'', '\"')
            dict_data = json.loads(data)
            extractions.append(dict_data)

    return extractions

def stream_output(content):
    '''
    模拟文本内容的流式返回
    :param content:
    :return:
    '''
    output = ""
    for char in content:
        output += char
        yield output  # 使用yield逐步返回当前输出
        time.sleep(0.05)  # 模拟处理时间


def write_dicts_to_txt(list: List[dict], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        for data in list:
            file.write(str(data) + "\n")
    file.close()


def write_list_to_txt(dataset: List, file_path: str):
    if not file_path.endswith(".txt"):
        raise TypeError("[Enlight] File Requirements Txt File Format.")

    with open(file_path, "w", encoding="utf-8") as file:
        for data in dataset:
            file.write(str(data) + "\n")
    file.close()


if __name__ == '__main__':
    path = 'prompts/filter_prompt_template.txt'
    res = load_content_from_txt(file_path=path)
    print(res)