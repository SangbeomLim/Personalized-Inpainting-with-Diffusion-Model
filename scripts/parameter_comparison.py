import torch
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


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


train_config = OmegaConf.load(f"/home/user/Paint-by-Example/configs/pretrain_text_inversion.yaml")
pretrain_model = load_model_from_config(train_config,
                                        f"/home/user/Paint-by-Example/experiments/caption_pretrain/2022-12-26T06-25-48_pretrain_text_inversion/checkpoints/epoch=000022.ckpt")

finetune_config = OmegaConf.load(f"/home/user/Paint-by-Example/configs/fine_tune.yaml")
finetune_model = load_model_from_config(finetune_config,
                                        f"/home/user/Paint-by-Example/experiments/fine_tune/inocent_bed/2023-01-02T08-52-50_fine_tune/checkpoints/epoch=000013.ckpt")

# print("Model's state_dict:")
# for param_tensor in pretrain_model.state_dict():
#     print(param_tensor, "\t", pretrain_model.state_dict()[param_tensor].size())

print("Parameter comparison (Module)")
parameter_dict = {}
params_count = {}
for pretrain_params, finetune_params in zip(pretrain_model.state_dict(), finetune_model.state_dict()):
    assert pretrain_params == finetune_params
    if pretrain_params.split(".")[0] not in parameter_dict:
        parameter_dict[pretrain_params.split(".")[0]] = 0
        params_count[pretrain_params.split(".")[0]] = 0

    parameter_dict[pretrain_params.split(".")[0]] += torch.mean(torch.abs(
        pretrain_model.state_dict()[pretrain_params] - finetune_model.state_dict()[finetune_params]).float())

    params_count[pretrain_params.split(".")[0]] += finetune_model.state_dict()[finetune_params].numel()

parameter_dict = {k: v for k, v in sorted(parameter_dict.items(), key=lambda item: item[1], reverse=True)}
params_count = {k: v for k, v in sorted(params_count.items(), key=lambda item: item[1], reverse=True)}
print(parameter_dict)
print(params_count)

print("Divided by Total Params")
for key in parameter_dict:
    parameter_dict[key] = parameter_dict[key] / params_count[key]
print(parameter_dict)

print("Parameter comparison (Layer), Diffusion Model")
parameter_dict = {}
params_count = {}
for name, param in pretrain_model.model.named_parameters():
    if name.split(".")[1] not in parameter_dict:
        parameter_dict[name.split(".")[1]] = 0
        params_count[name.split(".")[1]] = 0

    parameter_dict[name.split(".")[1]] += torch.mean(torch.abs(
        pretrain_model.model.state_dict()[name] - finetune_model.model.state_dict()[name]).float())
    params_count[name.split(".")[1]] += param.numel()

parameter_dict = {k: v for k, v in sorted(parameter_dict.items(), key=lambda item: item[1], reverse=True)}
params_count = {k: v for k, v in sorted(params_count.items(), key=lambda item: item[1], reverse=True)}
print(parameter_dict)
print(params_count)

print("Divided by Total Params")
for key in parameter_dict:
    parameter_dict[key] = parameter_dict[key] / params_count[key]
print(parameter_dict)

# for key in parameter_dict:
#     trainable_params = sum(p.numel() for p in pretrain_model.parameters() if p.requires_grad)
#     parameter_dict[key] = parameter_dict[key] / trainable_params