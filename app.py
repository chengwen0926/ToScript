import os
import sys
import argparse
import gradio as gr
import numpy as np
import random
# import torch
# import torchaudio
from pathlib import Path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


inference_mode_list = ['音频推理', '图像推理', '视频推理']
instruct_dict = {
    inference_mode_list[0]: '1. 选择\'{}\'并上传源音频\n2. 点击生成脚本按钮'.format(inference_mode_list[0]),
    inference_mode_list[1]: '1. 选择\'{}\'并上传源图像\n2. 点击生成脚本按钮'.format(inference_mode_list[1]),
    inference_mode_list[2]: '1. 选择\'{}\'并上传源视频\n2. 点击生成脚本按钮'.format(inference_mode_list[2])
}
stream_mode_list = [('是', True), ('否', False)]

def change_mode(mode_checkbox_group):
    aud_visible = {
        "__type__": "update",
        "visible": mode_checkbox_group==inference_mode_list[0]
    }
    img_visible = {
        "__type__": "update",
        "visible": mode_checkbox_group==inference_mode_list[1]
    }
    vid_visible = {
        "__type__": "update",
        "visible": mode_checkbox_group==inference_mode_list[2]
    }

    return instruct_dict[mode_checkbox_group] , aud_visible, img_visible, vid_visible

def generate_script():
    import time
    content = '《哪吒：魔童降世》讲述了一个颠覆传统的神话故事。天地灵气孕育出混元珠，元始天尊将其分为灵珠和魔丸，灵珠投胎为哪吒，魔丸则被施以天劫咒，三年后将被天雷摧毁。然而，灵珠被申公豹盗走，投胎为龙王之子敖丙，而哪吒则被误认为是魔丸转世。哪吒从小被视为妖怪，备受歧视，但他内心渴望被认可…'
    response = ''
    for char in content:
        response += char
        yield response  # 每次 yield 一个字符
        time.sleep(0.1)

def output_directly(para):
    return para

def main():
    theme = gr.themes.Soft()
    block = gr.Blocks(theme=theme).queue()
    with block as demo:
        gr.Markdown("### 代码库 [ToScript](https://github.com/chengwen0926/ToScript) \
                    预训练模型 [Whisper-Large-v3-Turbo](https://huggingface.co/openai/whisper-large-v3-turbo)")
        gr.Markdown("#### 请选择推理模式，并按照提示步骤进行操作")

        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0])
            stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][0])
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]])
        
        with gr.Accordion(label='模型及其他配置信息', open=False) as configuration:
            with gr.Row():
                with gr.Column():
                    gr.Radio(["Audio2Script", "Image2Script", "Video2Script"], value="Audio2Script", label='模型类型')
                with gr.Column():
                    gr.Textbox(placeholder='Input HuggingFace Repository ID to Download', label='🤗 Repo ID', interactive=True)
                with gr.Column():
                    gr.Button(value="模型下载&加载部署")

           

        with gr.Row() as aud_upload_row:
            aud_upload = gr.Audio(sources="upload", type='filepath', label='选择音频文件，注意采样率不低于16khz')
        with gr.Row(visible=False) as img_upload_row:
            img_upload = gr.Image(sources="upload", type='filepath', label='选择图像文件')
        with gr.Row(visible=False) as vid_upload_row:
            vid_upload = gr.Video(sources="upload", label='选择视频文件')
        

        generate_button = gr.Button("生成文本")
        script_output = gr.Textbox(label='检测到的脚本内容', lines=10, autoscroll=True, interactive=False)        
        mode_checkbox_group.change(fn=change_mode, inputs=[mode_checkbox_group], outputs=[instruction_text, aud_upload_row, img_upload_row, vid_upload_row])
        generate_button.click(fn=generate_script,inputs=[],outputs=[script_output])

        # Gradio中的Video对象作为输入时报错TypeError: argument of type 'bool' is not iterable
        # 更新 pip install pydantic==2.10.6
        vid_upload.upload(fn=output_directly,inputs=[vid_upload],outputs=[script_output])

    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='127.0.0.1', share=True, server_port=8090)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--port',
    #                     type=int,
    #                     default=7890)
    # parser.add_argument('--model_dir',
    #                     type=str,
    #                     default='pretrained_models/whisper-large-v3-turbo',
    #                     help='local path or modelscope repo id')
    # args = parser.parse_args()
    # try:
    #     cosyvoice = CosyVoice(args.model_dir)
    # except Exception:
    #     try:
    #         cosyvoice = CosyVoice2(args.model_dir)
    #     except Exception:
    #         raise TypeError('no valid model_type!')

    # sft_spk = cosyvoice.list_available_spks()
    # if len(sft_spk) == 0:
    #     sft_spk = ['']
    # prompt_sr = 16000
    # default_data = np.zeros(cosyvoice.sample_rate)
    main()