# coding: utf-8
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

base_path = Path(__file__).parent.absolute()
cache_dir = os.path.join(base_path, "model_cache")
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(cache_dir, "huggingface")
os.environ['MODELSCOPE_CACHE'] = os.path.join(cache_dir, "modelscope")
os.environ['TORCH_HOME'] = os.path.join(cache_dir, "torch")
os.makedirs(cache_dir, exist_ok=True)

import gradio as gr
import torch
from generator import Generator, MODEL_CONFIG
from PIL import Image

modelscope = False


COLOR_PRESETS = {
    "Blue": "#2563eb",
    "Black": "#000000",
    "Red": "#dc2626",
    "Orange": "#CC6600",
    "Yellow": "#BBBB00",
    "Green": "#16a34a",
    "Purple": "#9333ea"
}

custom_css = """
:root { --primary-500: #2563eb; --primary-600: #1d4ed8; }
.gradio-container { max-width: 100% !important; margin: 0 !important; padding: 10px !important; } 
.compact-group { padding: 0 !important; gap: 4px !important; }
footer { display: none !important; }
input[type='range'] { accent-color: var(--primary-500) !important; }
.selected { border-color: var(--primary-500) !important; }

button[aria-label='share'] { display: none !important; }
"""

VERSION_DICT_IMAGE = {
    'HelloMemeV5': 'v5',
    'HelloMemeV5b': 'v5b',
    'HelloMemeV5c': 'v5c',
}

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as app:
    gen = Generator(gpu_id=0, dtype=torch.float16, sr=True, pipeline_dict_len=1, modelscope=modelscope)

    with gr.Row():
        gr.Markdown('''
            <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 20px;">
                <div style="display: flex; flex-direction: column; align-items: flex-start;">
                    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 800; letter-spacing: -1px; color: #1a1a1a;">
                        🎨 Emoji <span style="color: #4f46e5;">Transfer</span>
                    </h1>
                </div>
                <div style="display: flex; align-items: center; gap: 8px; margin-top: 10px;">
                    <a href='https://github.com/HSQhere/emoji-transfer'><img src='https://img.shields.io/badge/GitHub-Code-blue'></a>
                    <a href='https://arxiv.org/pdf/2410.22901'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
                    <a href='https://github.com/HSQhere/emoji-transfer'><img src='https://img.shields.io/github/stars/HSQhere/emoji-transfer'></a>
                </div>
            </div>
        ''')
        theme_dropdown = gr.Dropdown(
            choices=list(COLOR_PRESETS.keys()),
            value="Blue",
            label="Theme Color",
            container=False,
            scale=0
        )

    with gr.Tab("Image Generation"):
        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                with gr.Row():
                    ref_img = gr.Image(type="pil", label="Reference", height=260)
                    drive_img = gr.Image(type="pil", label="Drive", height=260)

                with gr.Group(elem_classes="compact-group"):
                    with gr.Row():
                        checkpoint = gr.Dropdown(
                            choices=list(MODEL_CONFIG['sd15']['checkpoints'].keys()),
                            value=list(MODEL_CONFIG['sd15']['checkpoints'].keys())[1],
                            label="Checkpoint"
                        )
                        lora = gr.Dropdown(
                            choices=['None'] + list(MODEL_CONFIG['sd15']['loras'].keys()),
                            value="None",
                            label="LoRA"
                        )
                    with gr.Row():
                        lora_scale = gr.Slider(0.0, 5.0, 1.0, step=0.1, label="Lora Scale")
                        version = gr.Dropdown(choices=list(VERSION_DICT_IMAGE.keys()), value="HelloMemeV5",
                                              label="Version")
                        stylize = gr.Dropdown(choices=['x1', 'x2'], value="x1", label="Stylize")
                        cntrl_version = gr.Textbox(value="HMControlNet2", label="Control", interactive=False)

                exec_btn = gr.Button("🚀 Run Generation", variant="primary")

            with gr.Column(scale=3):

                result_img = gr.Image(
                    type="pil",
                    label="Generated Image",
                    format="jpeg",
                    interactive=False,
                    show_label=True,
                    height=560,
                    elem_id="result-img"
                )

        with gr.Accordion("⚙️ Advanced Options", open=False):
            with gr.Row():
                num_steps = gr.Slider(1, 50, 25, step=1, label="Steps")
                guidance = gr.Slider(1.0, 10.0, 1.5, step=0.1, label="Guidance")
                seed = gr.Number(value=-1, label="Seed", precision=0)
                trans_ratio = gr.Slider(0.0, 1.0, 0.0, step=0.01, label="Trans Ratio")
                crop_reference = gr.Checkbox(label="Crop Reference", value=True)

    theme_dropdown.change(None, inputs=theme_dropdown, outputs=None, js=f"""
        (selection) => {{
            const colors = {COLOR_PRESETS};
            const color = colors[selection];
            document.documentElement.style.setProperty('--primary-500', color);
            document.documentElement.style.setProperty('--primary-600', color);
            const btn = document.querySelector('button.primary');
            if (btn) {{
                btn.style.backgroundColor = color;
                btn.style.borderColor = color;
            }}
        }}
    """)


    def img_gen_fnc(ref_img, drive_img, num_steps, guidance, seed,
                    trans_ratio, crop_reference, version, stylize, checkpoint, lora, lora_scale):
        if ref_img is None or drive_img is None:
            return None

        if lora != 'None':
            tmp_lora_info = MODEL_CONFIG['sd15']['loras'][lora]
            from huggingface_hub import hf_hub_download
            lora_path = hf_hub_download(tmp_lora_info[0], filename=tmp_lora_info[1])
        else:
            lora_path = None

        checkpoint_path = MODEL_CONFIG['sd15']['checkpoints'][checkpoint]

        try:
            token = gen.load_pipeline(
                "image",
                checkpoint_path=checkpoint_path,
                lora_path=lora_path,
                lora_scale=lora_scale,
                stylize=stylize,
                version=VERSION_DICT_IMAGE[version]
            )
            res = gen.image_generate(
                token, ref_img, drive_img, int(num_steps), guidance, int(seed),
                '', '', trans_ratio, crop_reference, 'cntrl2'
            )

            if res:
                res = res.convert("RGB")
                res.load()
                return res
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None


    exec_btn.click(
        fn=img_gen_fnc,
        inputs=[ref_img, drive_img, num_steps, guidance, seed,
                trans_ratio, crop_reference, version, stylize, checkpoint,
                lora, lora_scale],
        outputs=result_img
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)