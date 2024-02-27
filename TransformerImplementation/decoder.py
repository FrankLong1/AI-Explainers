import torch
import torch.nn as nn
from self_attention import SelfAttention
from transformer_block import TransformerBlock

# Single Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        
        # Transformer block (including FFN)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        
        # Apply layer normalization and dropout
        query = self.dropout(self.norm(attention + x))
        
        # Pass through the transformer block
        out = self.transformer_block(value, key, query, src_mask)

        return out

# Decoder consisting of multiple decoder blocks
class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device

        # Word and positional embeddings
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        # Fully connected output layer
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        # Pass through each of the decoder layers
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        # Output layer
        out = self.fc_out(x)

        return out