import torch
import torch.nn as nn
from transformer_models import GPT
from torchsummary import summary

# NOTE do a visualization of the attention mechanism using the same heatmap stuff they are using here
# https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.number_to_word = {v: k for k, v in self.vocab.items()}
        
    def encode_sentence(self, sentence):
        words = sentence.split()  # Split the sentence into a list of words
        return [self.vocab[word] for word in words]

def train_gpt(model, training_data, tokenizer, device):
    # Set the model to training mode
    model.train()

    # encode all the sentences in the training data
    encoded_sentences = [tokenizer.encode_sentence(sentence) for sentence in training_data]
        
    # Create an optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # how many times I want to update the parameters
    num_epochs = 10
    
    for epoch in range(num_epochs):
        # Iterate over the encoded sentences and perform training
        for encoded_sentence in encoded_sentences:
            # Convert the encoded sentence to tensor
            x = torch.tensor(encoded_sentence).unsqueeze(0).to(device)

            # Forward pass
            logits = model(x)
            # Compute the loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), x.view(-1))
            print("Loss", loss)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()





if __name__ == "__main__":
    # Note "PAD", "START", "END" all got removed from the example
    vocab = {
        "Rudolph": 0,
        "the": 1,
        "red": 2,
        "nosed": 3,
        "reindeer": 4,
        "had": 5,
        "a": 6,
        "very": 7,
        "shiny": 8,
        "nose": 9
    }


    tokenizer = Tokenizer(vocab)
    # Size of the vocabulary, i.e., the number of unique tokens in the model's language.
    vocab_size = len(vocab)     # 50257 in GPT-3, 8 for this one
    # Size of the token embeddings. This is the size of the vector representing each token.
    embed_size = 512           # 12288 in GPT-3 (referred to as "dmodel" in GPT papers)
    # Number of attention heads in the multi-head self-attention mechanism.
    heads = 8                  # Example: 96 in GPT-3
    # Expansion factor for the feedforward neural networks within the transformer.
    forward_expansion = 4      # Not specified in GPT-3 paper
    # Dropout probability used to regularize the model during training.
    dropout = 0.1              # Not specified in GPT-3 paper
    # Maximum sequence length the model can handle. Sequences longer than this will be truncated.
    context_window_size = 100  # 4096 tokens in production ChatGPT
    # Number of transformer layers in the model, which stacks multiple transformer blocks.
    num_layers = 1             # 96 layers in GPT-3


    device = "cpu"
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
    print(device)

    # Instantiate the GPT model with the specified parameters and move it to the chosen device.
    model = GPT(
        vocab_size=vocab_size,
        embed_size=embed_size,
        heads=heads,
        forward_expansion=forward_expansion,
        dropout=dropout,
        device=device,
        context_window_size=context_window_size,
        num_layers=num_layers,
        pad_idx = 0
    ).to(device)

    summary(model=model, input_size=(100,))

    sentence = "Rudolph the red nosed reindeer"
    encoded_sentence = tokenizer.encode_sentence(sentence)

    # Convert the list to a tensor and add a batch dimension
    x = torch.tensor(encoded_sentence).unsqueeze(0).to(device)
    # Feed the sentence into the model
    out = model(x)
    
    # What comes out of the model is essentially a probability distribution of the next token
    # The shape is batch size, sequence length, vocab size -- it predicts the previous tokens in the distribution
    print(out.shape)

    # Get the model's prediction for the next word (you'll see it calculates for all the other words also)
    prediction = out.argmax(2)[:, -1].item()

    # Decode the prediction
    predicted_word = tokenizer.number_to_word[prediction]


    # "Rudolph the red nosed ________"
    print(f"The predicted next word is: {predicted_word}")
    
    # NOTE this output is totally random so far... do some training!
    training_data = [
        "Rudolph the red nosed reindeer", 
        "had a very shiny nose"
        ]
    train_gpt(model, training_data, tokenizer, device)

    new_out = model(x)
    prediction = new_out.argmax(2)[:, -1].item()
    predicted_word = tokenizer.number_to_word[prediction]
    print(f"After training, the predicted next word is: {predicted_word}")