import torch
import torch.nn as nn
from transformer_block import TransformerBlock

class GPTDecoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
            num_layers,
        ):

        super(GPTDecoder, self).__init__()
        self.device = device
        
        # Word and positional embeddings
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Model layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    norm_first=True
                )
                for _ in range(num_layers)
            ]
        )
    # We are now inside the Decoder for GPT, which consists of a series of transformer blocks
    def forward(self, x, mask):

        # N = the batch size, i.e. number of examples being processed at once
        # seq_length = the length of the sequence within this batch (max possible is the context window size)
        N, seq_length = x.shape
        
        # Generating positional encodings to combine with the word embeddings, this gives the model a notion of which part of the sentence particular tokens are placed in
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        # NOTE: ROUNDING HERE DOES NOT MAKE SENSE THIS IS JUST HACKY NONSENSE TO MAKE TORCHSUMMARY WORK.
        x = torch.round(x).to(torch.int64)

        # Here we transform our humble inidices into "embeddings", which are more complex representations of the tokens
        # The variable "out" consists of our processed data, it will eventually be turned into our final output
        out = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        # Pass through each of the transformer blocks (i.e. the layers) one by one
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out