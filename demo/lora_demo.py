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
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

UNET_DEFAULT_TARGET_REPLACE = {"CrossAttention", "Attention", "GEGLU"}

DEFAULT_TARGET_REPLACE = UNET_DEFAULT_TARGET_REPLACE

class LoraInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )

        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = 1.0

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.normal_(self.lora_up.weight, std=1 / r)
        # nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return self.linear(input) + self.lora_up(self.lora_down(input)) * self.scale

def _find_modules_v2(
    model,
    ancestor_class: Set[str] = DEFAULT_TARGET_REPLACE,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [LoraInjectedLinear],
):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """

    # Get the targets we should replace all linears under
    ancestors = (
        module
        for module in model.modules()
        if module.__class__.__name__ in ancestor_class
    )

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                # Otherwise, yield it
                yield parent, name, module

def monkeypatch_lora(
    model, loras, target_replace_module=DEFAULT_TARGET_REPLACE, r: int = 4
):
    for _module, name, _child_module in _find_modules_v2(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        _tmp = LoraInjectedLinear(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r=r,
        )
        _tmp.linear.weight = weight

        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _module._modules[name] = _tmp

        up_weight = loras.pop(0)
        down_weight = loras.pop(0)

        _module._modules[name].lora_up.weight = nn.Parameter(
            up_weight.type(weight.dtype)
        )
        _module._modules[name].lora_down.weight = nn.Parameter(
            down_weight.type(weight.dtype)
        )

        _module._modules[name].to(weight.device)

def tune_lora_scale(model, alpha: float = 1.0):
    for _module in model.modules():
        if _module.__class__.__name__ == "LoraInjectedLinear":
            _module.scale = alpha

pipe = DiffusionPipeline.from_pretrained(
    "Fantasy-Studio/Paint-by-Example",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda:5")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config) # Important

monkeypatch_lora(pipe.unet, torch.load("/home/user/lora/efficient/mefema_sofa/lora_weight_e6_s4000.pt"))
tune_lora_scale(pipe.unet, 1.00)

from share_btn import community_icon_html, loading_icon_html, share_js


def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


def predict(dict, reference, scale, seed, step):
    width, height = dict["image"].size
    if width < height:
        factor = width / 512.0
        width = 512
        height = int((height / factor) / 8.0) * 8

    else:
        factor = height / 512.0
        height = 512
        width = int((width / factor) / 8.0) * 8
    init_image = dict["image"].convert("RGB").resize((width, height))
    mask = dict["mask"].convert("RGB").resize((width, height))
    generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None
    output = pipe(
        image=init_image,
        mask_image=mask,
        example_image=reference,
        generator=generator,
        guidance_scale=scale,
        num_inference_steps=step,
    ).images[0]
    return output


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
example = {}
ref_dir = 'images/reference'
image_dir = 'images/source'
ref_list = [os.path.join(ref_dir, file) for file in os.listdir(ref_dir)]
ref_list.sort()
image_list = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
image_list.sort()

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
                    steps = gr.Slider(label="Steps", value=50, minimum=2, maximum=75, step=1, interactive=True)

                    seed = gr.Slider(0, 1000, label='Seed (0 = random)', value=500, step=1)

                    with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                        btn = gr.Button("Paint!").style(
                            margin=False,
                            rounded=(False, True, True, False),
                            full_width=True,
                        )
                    with gr.Group(elem_id="share-btn-container"):
                        community_icon = gr.HTML(community_icon_html, visible=True)
                        loading_icon = gr.HTML(loading_icon_html, visible=True)

            with gr.Row():
                with gr.Column():
                    gr.Examples(image_list, inputs=[image], label="Examples - Source Image", examples_per_page=12)
                with gr.Column():
                    gr.Examples(ref_list, inputs=[reference], label="Examples - Reference Image", examples_per_page=12)

            btn.click(fn=predict, inputs=[image, reference, guidance, seed, steps],
                      outputs=[image_out])

try:
    image_blocks.launch(server_port=8852, server_name="0.0.0.0", share=True)
except KeyboardInterrupt:
    image_blocks.close()
except Exception as e:
    print(e)
    image_blocks.close()
