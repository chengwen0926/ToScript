# ToScript 🎙️📷📽️->📄

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

## Clone and Installation

**Clone the Repository**

``` sh
git clone https://github.com/chengwen0926/ToScript.git
cd ToScript
```

**Create Execution Environment**

``` sh
conda create -n toscript -y python=3.10
conda activate toscript
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# Install torch, you can choose other torch versions through the following URLs: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

```

**Model download**
``` sh
mkdir pretrained_models
git lfs install
# Download Whisper-large-v3 turbo model
git clone https://huggingface.co/openai/whisper-large-v3-turbo
```


## Future Updates ✌️
- [ ] utils.py 脚本中有关文件处理代码需要进一步完善
- [ ] app.py 脚本中的参数功能补全argparse
- [ ] 检测模型文件是否存在，不存在的话就先执行下载操作
- [ ] ToScript类的封装
  - [ ] 参考cozyvoice的写法
