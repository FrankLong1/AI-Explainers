import torch
import torch.nn as nn

# Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads  # Dimension of each multi-head

        # Ensure that the embedding size is a multiple of the number of heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # Linear layers for the queries, keys and values
        # TODO: implement the ablity to fan out into higher dimmensionality to represent more complex attention
        self.values_layer = nn.Linear(embed_size, embed_size)
        self.keys_layer = nn.Linear(embed_size, embed_size)
        self.queries_layer = nn.Linear(embed_size, embed_size)
        
        # Fully Connected layer for the concatenated outputs
        self.fc_out = nn.Linear(embed_size, embed_size)

    # We are now within the famous attention mechanism!!!
    def forward(self, values, keys, query, mask):
        # Number of training examples (batch size)
        N = query.shape[0]

        # Sequence lengths for values, keys, queries
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = self.values_layer(values)  # (N, value_len, embed_size)
        keys = self.keys_layer(keys)  # (N, key_len, embed_size)
        query = self.queries_layer(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)


        # Scaled Dot-Product Attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])

        # Apply a mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Attention scores
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Weighted sum of the values based on the attention scores
        out = torch.einsum("nhqv,nvhd->nqhd", [attention, values])

        # Concatenating multi-head outputs
        out = out.reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)

        return out