import random

import gradio as gr

from io import BytesIO
import requests
import PIL
from PIL import Image
import numpy as np
import os
import uuid
import torch
from torch import autocast
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms
from diffusers import DiffusionPipeline
from diffusers.utils import torch_device
from ldm.models.diffusion.plms import PLMSSampler

from share_btn import community_icon_html, loading_icon_html, share_js
from functions import load_model_from_config, model_forward, basemodel_forward
from omegaconf import OmegaConf

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_path_list = [
    "/home/user/Paint-by-Example/experiments/fine_tune/fabric_sofa/2023-01-18T07-33-54_fine_tune/checkpoints/epoch=000001.ckpt",
    # "/home/user/Paint-by-Example/experiments/fine_tune/fabric_sofa/2023-01-18T07-48-08_fine_tune/checkpoints/epoch=000002.ckpt",
    "/home/user/Paint-by-Example/experiments/fine_tune/mask_augmentation/fabric_sofa/2023-01-18T07-34-10_mask_augmentation_fine_tune/checkpoints/epoch=000003.ckpt"
]
config_path_list = [
    "/home/user/Paint-by-Example/demo/model_config.yaml",
    "/home/user/Paint-by-Example/configs/fine_tune/mask_augmentation_fine_tune.yaml",
]

model_list = []
sampler_list = []
class_list = ['table', 'sofa', 'lamp', 'bed', 'sofa']
for config_path, model_path in zip(config_path_list, model_path_list):
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, model_path)
    model = model.to(device)
    model_list.append(model)
    sampler = PLMSSampler(model)
    sampler_list.append(sampler)

example = {}
ref_dir = 'images/reference'
image_dir = 'images/source'
ref_list = [os.path.join(ref_dir, file) for file in os.listdir(ref_dir)]
ref_list.sort()
image_list = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
image_list.sort()

cherry_picked_reference_list = ref_list = [os.path.join(ref_dir, file) for
                                           file in os.listdir(ref_dir)]
cherry_picked_reference_dict = {}
for i, ref in enumerate(cherry_picked_reference_list):
    ref_image = Image.open(ref).convert("RGB").resize((224, 224))
    cherry_picked_reference_dict[i] = ref_image


def predict(dict, reference, scale, seed, step):
    width, height = 512, 512
    if width < height:
        factor = width / 512.0
        width = 512
        height = int((height / factor) / 8.0) * 8
    else:
        factor = height / 512.0
        height = 512
        width = int((width / factor) / 8.0) * 8
    # width, height= 512, 512
    init_image = dict["image"].convert("RGB").resize((width, height))
    mask = dict["mask"].convert("L").resize((width, height))
    reference = reference.convert("RGB").resize((224, 224))
    seed = random.randint(0, 1000) if seed == 0 else seed

    # index = list(cherry_picked_reference_dict.keys())[list(cherry_picked_reference_dict.values()).index(reference)]
    index=0

    if index not in [0]:
        output = basemodel_forward(
            image=init_image,
            mask_image=mask,
            reference_image=reference,
            seed=seed,
            guidance_scale=scale,
            ddim_step=step,
            model=model_list[index],
            sampler=sampler_list[index],
            device=device,
        )
    else:
        output = model_forward(
            image=init_image,
            mask_image=mask,
            reference_image=reference,
            seed=seed,
            guidance_scale=scale,
            ddim_step=step,
            model=model_list[index],
            sampler=sampler_list[index],
            device=device,
            class_type=class_list[index],
        )
    return output  # , gr.update(visible=True), gr.update(visible=True)


def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


css = '''
.container {max-width: 1150px;margin: auto;padding-top: 1.5rem}
#image_upload{min-height:400px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 400px}
#mask_radio .gr-form{background:transparent; border: none}
#word_mask{margin-top: .75em !important}
#word_mask textarea:disabled{opacity: 0.3}
.footer {margin-bottom: 45px;margin-top: 35px;text-align: center;border-bottom: 1px solid #e5e5e5}
.footer>p {font-size: .8rem; display: inline-block; padding: 0 10px;transform: translateY(10px);background: white}
.dark .footer {border-color: #303030}
.dark .footer>p {background: #0b0f19}
.acknowledgments h4{margin: 1.25em 0 .25em 0;font-weight: bold;font-size: 115%}
#image_upload .touch-none{display: flex}
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
#share-btn-container {
    display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
}
#share-btn {
    all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
}
#share-btn * {
    all: unset;
}
#share-btn-container div:nth-child(-n+2){
    width: auto !important;
    min-height: 0px !important;
}
#share-btn-container .wrap {
    display: none !important;
}
'''

image_blocks = gr.Blocks(css=css)
with image_blocks as demo:
    gr.HTML(read_content("header.html"))
    with gr.Group():
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    image = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil",
                                     label="Source Image")
                    reference = gr.Image(source='upload', elem_id="image_upload", type="pil", label="Reference Image")

                with gr.Column():
                    image_out = gr.Image(label="Output", elem_id="output-img").style(height=400)
                    guidance = gr.Slider(label="Guidance scale", value=5, maximum=15, interactive=True)
                    steps = gr.Slider(label="Steps", value=50, minimum=2, maximum=100, step=1, interactive=True)
                    seed = gr.Slider(0, 1000, label='Seed (0 = random)', value=500, step=1)

                    with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                        btn = gr.Button("Paint!").style(
                            margin=False,
                            rounded=(False, True, True, False),
                            full_width=True,
                        )
                    with gr.Group(elem_id="Naver-btn-container"):
                        community_icon = gr.HTML(community_icon_html, visible=True)
                        loading_icon = gr.HTML(loading_icon_html, visible=True)

            with gr.Row():
                with gr.Column():
                    gr.Examples(image_list, inputs=[image], label="Examples - Source Image", examples_per_page=12)
                with gr.Column():
                    gr.Examples(ref_list, inputs=[reference], label="Examples - Reference Image (Select One)",
                                examples_per_page=12)

            btn.click(fn=predict, inputs=[image, reference, guidance, seed, steps],
                      outputs=[image_out])

try:
    image_blocks.launch(server_port=8852, server_name="0.0.0.0", share=True)
except KeyboardInterrupt:
    image_blocks.close()
except Exception as e:
    print(e)
    image_blocks.close()
