# 创建 pretrained_models 目录（如果不存在）
New-Item -Path "pretrained_models" -ItemType Directory -Force

# 进入该目录
Set-Location -Path "pretrained_models"

# 初始化 Git LFS
git lfs install

# 克隆指定模型仓库
git clone https://huggingface.co/openai/whisper-large-v3-turbo

# 可选：返回原始目录（根据需求）
Set-Location ..