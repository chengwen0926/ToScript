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


# 路径相关操作
def check_dir(path:str, mkdir:bool=True)->bool:
    '''
        检查路径或者文件是否存在，若不存在可以指定是否创建
    Args:
        path: 目录或文件路径
        mkdir: 在目录或文件路径不存在的情况下是否创建
    Returns:
        输入目录是否存在，存在则返回True，不存在则返回False
    '''
    # 已存在路径
    dir_path = Path(path)
    if dir_path.exists():
        logging.info(f"[Enlight] 目录已存在：{dir_path}")
        return True
    
    # 创建此前不存在的文件
    elif is_likely_file(path) and mkdir:
        parent_dir = os.path.dirname(path)
        os.makedirs(parent_dir, exist_ok=True) # 父目录若不存在则递归创建
        with open(path, 'w', encoding='utf-8') as f:
            f.close()
    
    # 创建此前不存在的路径
    elif mkdir:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"[Enlight] 目录创建成功：{dir_path}")
        except PermissionError:
            logging.info(f"[Enlight] 权限不足，无法创建目录：{dir_path}")
        except Exception as e:
            logging.error(f"[Enlight] 目录创建失败: {str(e)}")
        
    return False
    


# 文件处理相关操作

def is_likely_file(path:str)->bool:
    filename = os.path.basename(path)
    return '.' in filename and len(filename.split('.')[-1]) > 0


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


def append_str_to_file(content: str, file_path: str) -> bool:
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
    tmp = './outputs/test.txt'
    res = check_dir(tmp)
    print(res)