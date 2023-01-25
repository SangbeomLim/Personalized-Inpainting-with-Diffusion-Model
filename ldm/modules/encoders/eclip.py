from collections import OrderedDict
from collections.abc import Iterable
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoints = int(self.layers ** 0.5 + 0.5)

    def forward(self, x: torch.Tensor):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint_sequential(self.resblocks, self.checkpoints, x)

        return self.resblocks(x)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False


class VisualTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.gradient_checkpointing = gradient_checkpointing
        self.transformer = Transformer(width, layers, heads,
                                       gradient_checkpointing=gradient_checkpointing)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim)) # outdim = 512

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, channel, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, channel, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, channel]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width] CLass token added
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # [1,50,768]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # print(x.shape)
        x = self.ln_post(x[:, 0, :]) # CLS POOLing
        if self.proj is not None: # 768 -> 512
            x = x @ self.proj
        # print(x.shape) # [1,512]
        return x

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        self.transformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.transformer.gradient_checkpointing_disable()


class TextTransformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 context_length: int,
                 vocab_size: int,
                 output_dim: int,
                 gradient_checkpointing: bool = False,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vocab_size = vocab_size

        self.gradient_checkpointing = gradient_checkpointing
        self.transformer = Transformer(width, layers, heads,
                                       attn_mask=self.build_attention_mask(),
                                       gradient_checkpointing=gradient_checkpointing)

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(torch.empty(width, output_dim))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text: torch.Tensor):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        self.transformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.transformer.gradient_checkpointing_disable()


class PoolingLayer(nn.Module):
    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 ):
        super(PoolingLayer, self).__init__()

        self.config_keys = ['word_embedding_dimension', 'pooling_mode_cls_token', 'pooling_mode_mean_tokens',
                            'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens

        pooling_mode_multiplier = sum([pooling_mode_cls_token,
                                       pooling_mode_max_tokens,
                                       pooling_mode_mean_tokens,
                                       pooling_mode_mean_sqrt_len_tokens])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def forward(self, token_embeddings, cls_token_embeddings, attention_mask, token_weights_sum=None):
        # Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token_embeddings)

        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)

        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            if token_weights_sum is not None:
                sum_mask = token_weights_sum.unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        return output_vector

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return PoolingLayer(**config)


class ECLIP(nn.Module):
    def __init__(self,
                 image_model,
                 text_model,
                 output_dim: int,
                 image_dim: int,
                 text_dim: int,
                 ):
        super().__init__()
        self.image_model = image_model
        self.text_model = text_model

        # Image
        self.image_classifier = nn.Linear(image_dim, output_dim)

        # Text
        self.pooling = PoolingLayer(text_dim,
                                    pooling_mode_mean_tokens=True,
                                    pooling_mode_cls_token=False,
                                    pooling_mode_max_tokens=False)
        text_dim = self.pooling.pooling_output_dimension
        self.text_classifier = nn.Parameter(torch.empty(text_dim, output_dim))
        # Initialize parameters
        nn.init.normal_(self.text_classifier, std=text_dim ** -0.5)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        image_features = self.image_model(image)
        image_features = self.image_classifier(image_features) # Essential?
        return image_features

    def encode_text(self, text):
        if isinstance(text, dict) and 'input_ids' in text:
            text = text['input_ids']

        attention_mask = (text != 0).int()

        text_features = self.text_model(text)
        if isinstance(text_features, Iterable):
            text_features = text_features[0]

        text_features = self.pooling(
            token_embeddings=text_features,
            cls_token_embeddings=None,
            attention_mask=attention_mask,
        )
        text_features = text_features @ self.text_classifier

        return text_features

    def forward(self, image, text, label=None):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp().mean()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        outputs = (logits_per_image, logits_per_text,)

        if label is not None:
            loss_image = F.cross_entropy(logits_per_image, label)
            loss_text = F.cross_entropy(logits_per_text, label)
            loss = (loss_image + loss_text) / 2
            outputs += (loss,)

        return outputs


def load_image_model(model_path):
    state_dict = torch.load(model_path, map_location="cpu")

    if not isinstance(state_dict, dict):
        state_dict = {x[0]: x[1] for x in state_dict.named_parameters()}

    vision_width = state_dict["image_model.conv1.weight"].shape[0]
    vision_layers = len(
        [k for k in state_dict.keys() if k.startswith("image_model.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["image_model.conv1.weight"].shape[-1]
    grid_size = round((state_dict["image_model.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    # print(image_resolution)
    embed_dim = state_dict['image_model.proj'].shape[1]

    vision_heads = vision_width // 64
    image_model = VisualTransformer(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_heads,
        output_dim=embed_dim,
    )
    return image_model, embed_dim


def load_model(load_path):
    checkpoint = torch.load(load_path, map_location='cpu')  # GPU Usage?
    output_dim = checkpoint["text_classifier"].shape[1]

    image_model, image_dim = load_image_model(load_path)
    text_model = BertModel.from_pretrained('bert-base-multilingual-uncased', add_pooling_layer=False)
    text_dim = text_model.config.hidden_size

    model = ECLIP(image_model, text_model, output_dim, image_dim, text_dim)
    model.load_state_dict(checkpoint)

    return model


if __name__ == "__main__":
    model_path = "/home/user/image_editing/pretrained_models/eclip_hard.pt"
    input_image=torch.randn([1,3,224,224])
    model, image_embedding_dim = load_image_model(model_path)

    output=model.forward(input_image)
    print(output.shape)


