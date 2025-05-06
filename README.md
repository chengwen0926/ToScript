# ToScript 🎙️📷📽️->📄

## Clone and Installation

**Clone the Repository**

``` sh
git clone https://github.com/chengwen0926/ToScript.git
cd ToScript
# If you failed to clone submodule due to network failures, please run following command until success
git submodule update --init --recursive
```

**Create Execution Environment**

``` sh
conda create -n toscript -y python=3.10
conda activate toscript
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# Install torch, you can choose other torch versions through the following URLs: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

```

**Model download**
``` sh
mkdir pretrained_models
git lfs install
git clone https://huggingface.co/openai/whisper-large-v3-turbo
```


## Future Updates ✌️
- [ ] app.py 脚本中的参数功能补全argparse
- [ ] 检测模型文件是否存在，不存在的话就先执行下载操作
- [ ] ToScript类的封装
  - [ ] 参考cozyvoice的写法
