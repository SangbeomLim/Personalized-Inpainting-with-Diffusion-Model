# from transformers import ViTImageProessor, ViTModel
from PIL import Image
import requests
from transformers import AutoFeatureExtractor, AutoModel
import torch.nn as nn
from ldm.data.lsun import LSUNDataset, get_tensor, get_tensor_clip
import matplotlib.pyplot as plt
import argparse, os, sys, glob
import random
from datetime import datetime

random.seed(datetime.now().timestamp())
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import clip
from torchvision.transforms import Resize
from scripts.inference import get_class_label, check_safety
from tqdm.auto import tqdm
import torchvision.transforms.functional as TF

config_file = "/home/user/Paint-by-Example/configs/class_label.yaml"
model_path = "/home/user/Paint-by-Example/experiments/class_label/2022-12-16T18-21-03_class_label/checkpoints/epoch=000035.ckpt"
# model_path = "/home/user/Paint-by-Example/experiments/lsun/CLIP/checkpoints/best.ckpt"
# config_file = "/home/user/Paint-by-Example/configs/v1.yaml"

class_label_exist = True


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")

    pl_sd = torch.load(ckpt, map_location="cpu")  # CPU
    # pl_sd = torch.load(ckpt, map_location="cuda") # GPU
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def resize_img_label(bbox, original_size, target_size):
    bbox[0] = int(bbox[0] * target_size[1] / original_size[1])
    bbox[2] = int(bbox[2] * target_size[1] / original_size[1])
    bbox[1] = int(bbox[1] * target_size[0] / original_size[0])
    bbox[3] = int(bbox[3] * target_size[0] / original_size[0])

    return bbox


def un_norm(x):
    return (x + 1.0) / 2.0


def un_norm_clip(x):
    x[0, :, :] = x[0, :, :] * 0.26862954 + 0.48145466
    x[1, :, :] = x[1, :, :] * 0.26130258 + 0.4578275
    x[2, :, :] = x[2, :, :] * 0.27577711 + 0.40821073
    return x


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--plms",
    action='store_true',
    help="use plms sampling",
)
parser.add_argument(
    "--fixed_code",
    action='store_true',
    help="if enabled, uses the same starting code across samples ",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=2,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=1,
    help="how many samples to produce for each given reference image. A.k.a. batch size",
)
parser.add_argument(
    "--n_imgs",
    type=int,
    default=100,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--scale",
    type=float,
    default=5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--seed_type",
    type=str,
    default="None",
    help="choose seed type from given and random"
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--gpu",
    type=str,
    help="Usage of GPU",
    default="true"
)
parser.add_argument(
    "--class_label",
    type=str,
    help="input class type in string type",
    default=None
)
parser.add_argument(
    "--precision",
    type=str,
    default="autocast"
)
parser.add_argument(
    "--eval_num",
    type=int,
    help="number of testset to evaluate",
    default=100
)
opt = parser.parse_args()

testset = LSUNDataset(state="test", dataset_dir="/home/user/Paint-by-Example/dataset/lsun", arbitrary_mask_percent=0.5,
                      image_size=512,
                      class_label_exist=class_label_exist)

config = OmegaConf.load(f"{config_file}")
model = load_model_from_config(config, f"{model_path}")

if opt.gpu == "true":
    gpu_num = random.randint(0, 5)
    device = torch.device("cuda:" + str(gpu_num)) if torch.cuda.is_available() else torch.device("cpu")
else:
    device = torch.device("cpu")
model = model.to(device)
model.eval()

if opt.plms:
    sampler = PLMSSampler(model)
else:
    sampler = DDIMSampler(model)

start_code = None
if opt.fixed_code:
    start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

precision_scope = autocast if opt.precision == "autocast" else nullcontext

total_cosine_metric = 0
with torch.no_grad():
    with precision_scope("cuda"):
        with model.ema_scope():
            for i in tqdm(range(opt.eval_num)):
                # img_p, ref_p, mask_p, class_label, bbox = testset.test_get_item(i, None).values()
                img_p, _, mask_p, class_label, bbox = testset.test_get_item(i, None).values()
                _, ref_p, _, _, _ = testset.test_get_item(random.randint(0,300000), None).values()
                img_p_origin_size = img_p.shape[:2]
                img_p = Image.fromarray(img_p).resize((512, 512))
                image_tensor = get_tensor()(img_p)
                image_tensor = image_tensor.unsqueeze(0)

                ref_p = Image.fromarray(ref_p).resize((224, 224))
                ref_tensor = get_tensor_clip()(ref_p)
                ref_tensor = ref_tensor.unsqueeze(0)

                mask = mask_p.convert("L").resize((512, 512))
                mask = np.array(mask)[None, None]
                # mask = 1 - mask.astype(np.float32) / 255.0
                mask[mask < 0.5] = 0
                mask[mask >= 0.5] = 1
                mask_tensor = torch.from_numpy(mask)

                inpaint_image = image_tensor * mask_tensor
                test_model_kwargs = {}
                test_model_kwargs['inpaint_mask'] = mask_tensor.to(device)
                test_model_kwargs['inpaint_image'] = inpaint_image.to(device)
                ref_tensor = ref_tensor.to(device)
                uc = None
                if opt.scale != 1.0:
                    uc = model.learnable_vector
                c = model.get_learned_conditioning(ref_tensor.to(torch.float16))

                if class_label != 0:
                    class_label = torch.tensor([class_label]).to(device)
                    class_vector = model.class_embedding(class_label).unsqueeze(1)
                    c = torch.cat([c, class_vector], dim=-1)

                c = model.proj_out(c)
                inpaint_mask = test_model_kwargs['inpaint_mask']
                z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                test_model_kwargs['inpaint_image'] = z_inpaint
                test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-1], z_inpaint.shape[-1]])(
                    test_model_kwargs['inpaint_mask'])

                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 x_T=start_code,
                                                 test_model_kwargs=test_model_kwargs)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                x_checked_image = x_samples_ddim
                x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                for o, x_sample in enumerate(x_checked_image_torch):
                    all_img = []
                    all_img.append(un_norm(image_tensor[o]).cpu())
                    all_img.append(un_norm(inpaint_image[o]).cpu())
                    ref_img = ref_tensor
                    ref_img = Resize([512, 512])(ref_img)
                    all_img.append(un_norm_clip(ref_img[o]).cpu())
                    all_img.append(x_sample)
                    grid = torch.stack(all_img, 0)
                    grid = make_grid(grid)
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img.save("/home/user/Paint-by-Example/results/testset_sample/text_embedding/"+str(i)+".png")
