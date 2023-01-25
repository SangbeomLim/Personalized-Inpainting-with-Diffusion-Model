import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPModel, CLIPProcessor
import kornia
from ldm.modules.x_transformer import Encoder, \
    TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a requirement? --> test
from .xf import LayerNorm, Transformer
# from xf import LayerNorm, Transformer # For internal Execution, Uncomment when run main
import math
from PIL import Image
import requests
from ldm.modules.encoders.eclip import VisualTransformer, load_image_model


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda", use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  # .to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCopyCLIPEmbedder(AbstractEncoder):
    def __init__(self):
        super().__init__()
        self.text_model = FrozenCLIPTextEmbedder()
        self.image_model = FrozenCLIPImageEmbedder(cls_unpool=True)
        self.linear = nn.Linear(1024, 768)

        self.freeze()

    def freeze(self):
        self.text_model.freeze()
        self.image_model.freeze()

    def forward(self, text, image):
        image = self.image_model.forward(image)
        image = self.linear(image)
        text = self.text_model.forward(text)
        outputs = torch.cat([text, image], dim=1)  # [batch_size, mage+text token length, image & text dimension]

        return outputs

    def encode(self, text, image):
        return self(text, image)


class FrozenCLIPEmbedder(AbstractEncoder):
    def __init__(self):
        super().__init__()
        self.text_model = FrozenCLIPTextEmbedder()
        self.image_model = FrozenCLIPImageEmbedder()

        self.freeze()

    def freeze(self):
        self.text_model.freeze()
        # self.image_model.freeze()

    def forward(self, text, image):
        image = self.image_model.forward(image)
        text = self.text_model.forward(text)
        outputs = torch.cat([text, image], dim=-1)  # [batch_size, 1, image+text dimension]

        return outputs

    def encode(self, text, image):
        return self(text, image)


class FrozenCLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="openai/clip-vit-large-patch14", cls_unpool=None):
        super().__init__()
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.final_ln = LayerNorm(1024)
        self.mapper = Transformer(
            1,
            1024,
            5,
            1,
        )
        self.cls_unpool = cls_unpool

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True

    def forward(self, image):
        outputs = self.transformer(pixel_values=image)
        if self.cls_unpool != None:
            z = outputs.last_hidden_state
        else:
            z = outputs.pooler_output  # Pool EOS State, NOT SOS
            z = z.unsqueeze(1)
        # print(z.shape)print(z.shape)
        z = self.mapper(z)
        z = self.final_ln(z)
        return z

    def encode(self, image):
        return self(image)


class FrozenECLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="/home/user/image_editing/pretrained_models/eclip.pt"):
        super().__init__()
        model, image_embedding_dim = load_image_model(version)
        self.transformer = model
        self.final_ln = LayerNorm(512)
        self.mapper = Transformer(
            1,
            512,
            5,
            1,
        )

        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True

    def forward(self, image):
        outputs = self.transformer(image)
        # z = outputs.pooler_output # Pool CLS State
        z = outputs.unsqueeze(1)
        # print(z.shape)
        z = self.mapper(z)
        z = self.final_ln(z)
        return z

    def encode(self, image):
        return self(image)


class FrozenCLIPTextEmbedderMLP(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version).to(device)
        self.device = device
        self.max_length = max_length

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        outputs = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                 padding="max_length", return_tensors="pt").to(self.device)
        outputs = self.transformer(**outputs)
        z = outputs.pooler_output  # Pool CLS State
        z = z.unsqueeze(1)

        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPTextEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version).to(device)
        self.final_ln = LayerNorm(768)
        self.mapper = Transformer(
            1,
            768,
            5,
            1,
        )
        self.device = device
        self.max_length = max_length

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True

    def forward(self, text):
        outputs = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                 padding="max_length", return_tensors="pt").to(self.device)
        outputs = self.transformer(**outputs)
        z = outputs.pooler_output  # Pool CLS State
        z = z.unsqueeze(1)
        z = self.mapper(z)
        z = self.final_ln(z)
        return z

    def encode(self, text):
        return self(text)


if __name__ == "__main__":
    from ldm.util import count_params

    model = FrozenCLIPImageEmbedder()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    # model1 = FrozeneCLIPImageEmbedder()
    # count_params(model, verbose=True)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(images=image, return_tensors="pt")
    print(inputs.pixel_values.shape)

    outputs = model(inputs.pixel_values)
    print(outputs.shape)
    # last_hidden_state = outputs.last_hidden_state
