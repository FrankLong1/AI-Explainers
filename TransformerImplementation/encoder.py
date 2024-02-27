import torch
import torch.nn as nn

# Encoder consisting of multiple transformer blocks
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        # Word and positional embeddings
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        
        # Generating positional encodings
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        # Sum of word and positional embeddings
        out = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        # Pass through each of the transformer blocks
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out