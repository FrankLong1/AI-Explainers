import torch
import torch.nn as nn
from self_attention import SelfAttention

# Single Transformer Block (Encoder or Decoder)
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, norm_first=False):
        super(TransformerBlock, self).__init__()        
        # Optional Layer Norm at start
        self.norm_first = nn.LayerNorm(embed_size) if norm_first else None
        self.attention = SelfAttention(embed_size, heads)
        
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        # Feed-forward neural network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    # We are now within the famed "Transformer" block!
    def forward(self, value, key, query, mask):

        # If we're doing the GPT style we normalize first
        if self.norm_first:
            value = self.norm_first(value)
            key = self.norm_first(key)
            query = self.norm_first(query)

        # Here we enter the self-attention mechanism
        attention = self.attention(value, key, query, mask)
        
        # Layer normalization and dropout for the sum of the input and attention output
        x = self.dropout(self.norm1(attention + query))
        
        # Feed-forward neural network
        forward = self.feed_forward(x)
        
        # Layer normalization and dropout for the sum of the input and FFN output
        out = self.dropout(self.norm2(forward + x))

        return out
