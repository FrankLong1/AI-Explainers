import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder
from gpt_decoder import GPTDecoder



# Abstract Base Class
class BaseTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, context_window_size):
        super(BaseTransformer, self).__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.heads = heads
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        self.device = device
        self.context_window_size = context_window_size

    @staticmethod
    def make_mask(x, pad_idx):
        return (x != pad_idx).unsqueeze(1).unsqueeze(2).to(x.device)

    def forward(self):
        raise NotImplementedError

# Transformer Model (Encoder-Decoder)
class OriginalTransformer(BaseTransformer):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, **kwargs):
        super(OriginalTransformer, self).__init__(**kwargs)
        self.encoder = Encoder(
            src_vocab_size,
            self.embed_size,
            self.num_layers,
            self.heads,
            self.device,
            self.forward_expansion,
            self.dropout,
            self.context_window_size
        )
        self.decoder = Decoder(
            trg_vocab_size,
            self.embed_size,
            self.num_layers,
            self.heads,
            self.device,
            self.forward_expansion,
            self.dropout,
            self.context_window_size
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        return self.make_mask(src, self.src_pad_idx)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

# GPT Model
class GPT(BaseTransformer):
    def __init__(self, vocab_size, pad_idx, **kwargs):
        super(GPT, self).__init__(vocab_size=vocab_size, **kwargs)
        self.decoder = GPTDecoder(
            vocab_size,
            self.embed_size,
            self.heads,
            self.forward_expansion,
            self.dropout,
            self.device,
            self.context_window_size,
            self.num_layers,
        )
        self.fc_out = nn.Linear(self.embed_size, vocab_size)
        self.pad_idx = pad_idx

    def make_mask(self, x):
        return BaseTransformer.make_mask(x, self.pad_idx)

    # This is "Inference" aka the "forward pass" of the model
    def forward(self, x):
        # The masking logic is used for training to ensure that model does not "cheat" and see the future tokens
        mask = self.make_mask(x)
        # The GPT model is an "autoregressive transformer", meaning it is decoder only and consists of a bunch of transformer blocks stacked together
        x = self.decoder(x, mask)
        # Logits are the raw numbers that come out of the model, we convert those to probabilities using a "Softmax function"
        logits = self.fc_out(x)
        probabilities = F.softmax(logits, dim =-1)
        # The model returns the probabilities of each token being the correct next token
        return probabilities