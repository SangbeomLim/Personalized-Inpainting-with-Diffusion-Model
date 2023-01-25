
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
from torchvision.transforms import Resize, ToPILImage
import matplotlib.pyplot as plt


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


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


def model_forward(image, mask_image, reference_image, seed, guidance_scale, ddim_step, model, sampler, device,
                  class_type):
    # generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None
    seed_everything(seed)

    image_tensor = get_tensor()(image)
    image_tensor = image_tensor.unsqueeze(0)
    ref_tensor = get_tensor_clip()(reference_image)
    ref_tensor = ref_tensor.unsqueeze(0)
    mask = np.array(mask_image)[None, None]
    mask = 1 - mask.astype(np.float32) / 255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask_tensor = torch.from_numpy(mask)
    inpaint_image = image_tensor * mask_tensor

    test_model_kwargs = {}
    test_model_kwargs['inpaint_mask'] = mask_tensor.to(device)
    test_model_kwargs['inpaint_image'] = inpaint_image.to(device)
    ref_tensor = ref_tensor.to(device)
    uc = None
    c = model.get_learned_conditioning("a photo of a * " + str(class_type),
                                       ref_tensor.to(torch.float32))  # a photo of a

    c = model.proj_out(c)
    inpaint_mask = test_model_kwargs['inpaint_mask']
    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
    test_model_kwargs['inpaint_image'] = z_inpaint
    test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-1], z_inpaint.shape[-1]])(
        test_model_kwargs['inpaint_mask'])

    shape = [4, 512 // 8, 512 // 8]
    samples_ddim, _ = sampler.sample(S=ddim_step,
                                     conditioning=c,
                                     batch_size=1,
                                     shape=shape,
                                     verbose=False,
                                     unconditional_guidance_scale=guidance_scale,
                                     unconditional_conditioning=uc,
                                     eta=0.0,
                                     x_T=None,
                                     test_model_kwargs=test_model_kwargs)

    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

    x_samples_ddim_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

    x_sample = 255. * rearrange(x_samples_ddim_torch[0].cpu().numpy(), 'c h w -> h w c')
    return_img = Image.fromarray(x_sample.astype(np.uint8))

    def un_norm(x):
        return (x + 1.0) / 2.0

    def un_norm_clip(x):
        x[0, :, :] = x[0, :, :] * 0.26862954 + 0.48145466
        x[1, :, :] = x[1, :, :] * 0.26130258 + 0.4578275
        x[2, :, :] = x[2, :, :] * 0.27577711 + 0.40821073
        return x

    all_img = []
    all_img.append(un_norm(image_tensor[0]).cpu())
    all_img.append(un_norm(inpaint_image[0]).cpu())
    ref_img = ref_tensor
    ref_img = Resize([512, 512])(ref_img)
    all_img.append(un_norm_clip(ref_img[0]).cpu())
    all_img.append(x_samples_ddim_torch[0])
    grid = torch.stack(all_img, 0)
    grid = make_grid(grid)
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    img = Image.fromarray(grid.astype(np.uint8))
    img.save("/home/user/Paint-by-Example/demo/grid.png")

    return return_img

def basemodel_forward(image, mask_image, reference_image, seed, guidance_scale, ddim_step, model, sampler, device):
    seed_everything(seed)

    image_tensor = get_tensor()(image)
    image_tensor = image_tensor.unsqueeze(0)
    ref_tensor = get_tensor_clip()(reference_image)
    ref_tensor = ref_tensor.unsqueeze(0)
    mask = np.array(mask_image)[None, None]
    mask = 1 - mask.astype(np.float32) / 255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask_tensor = torch.from_numpy(mask)
    inpaint_image = image_tensor * mask_tensor

    test_model_kwargs = {}
    test_model_kwargs['inpaint_mask'] = mask_tensor.to(device)
    test_model_kwargs['inpaint_image'] = inpaint_image.to(device)
    ref_tensor = ref_tensor.to(device)
    uc = None
    c = model.get_learned_conditioning(ref_tensor.to(torch.float32))

    c = model.proj_out(c)
    inpaint_mask = test_model_kwargs['inpaint_mask']
    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
    test_model_kwargs['inpaint_image'] = z_inpaint
    test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-1], z_inpaint.shape[-1]])(
        test_model_kwargs['inpaint_mask'])

    shape = [4, 512 // 8, 512 // 8]
    samples_ddim, _ = sampler.sample(S=ddim_step,
                                     conditioning=c,
                                     batch_size=1,
                                     shape=shape,
                                     verbose=False,
                                     unconditional_guidance_scale=guidance_scale,
                                     unconditional_conditioning=uc,
                                     eta=0.0,
                                     x_T=None,
                                     test_model_kwargs=test_model_kwargs)

    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

    x_samples_ddim_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

    x_sample = 255. * rearrange(x_samples_ddim_torch[0].cpu().numpy(), 'c h w -> h w c')
    return_img = Image.fromarray(x_sample.astype(np.uint8))

    def un_norm(x):
        return (x + 1.0) / 2.0

    def un_norm_clip(x):
        x[0, :, :] = x[0, :, :] * 0.26862954 + 0.48145466
        x[1, :, :] = x[1, :, :] * 0.26130258 + 0.4578275
        x[2, :, :] = x[2, :, :] * 0.27577711 + 0.40821073
        return x

    all_img = []
    all_img.append(un_norm(image_tensor[0]).cpu())
    all_img.append(un_norm(inpaint_image[0]).cpu())
    ref_img = ref_tensor
    ref_img = Resize([512, 512])(ref_img)
    all_img.append(un_norm_clip(ref_img[0]).cpu())
    all_img.append(x_samples_ddim_torch[0])
    grid = torch.stack(all_img, 0)
    grid = make_grid(grid)
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    img = Image.fromarray(grid.astype(np.uint8))
    img.save("/home/user/Paint-by-Example/demo/logs/"+str(time.ctime()).replace(' ','').replace(":",'')+".png")

    return return_img