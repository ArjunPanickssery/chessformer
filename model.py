from os import name
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Parameters
input_dim = 4  # Input dimension (size of one-hot token)
output_dim = 16  # Output dimension (size of output one-hot vector)
d_model = 128  # Dimension of the model
nhead = 1  # Number of heads in the multiheadattention models
dim_feedforward = 128  # Dimension of the feedforward network model
num_tokens = 25  # Number of tokens in the sequence
batch_size = 32  # Batch size for training
num_batches = 100  # Number of batches for training
learning_rate = 0.001  # Learning rate for the optimizer
num_epochs = 5


class CustomDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        output_sample = self.output_data[idx]
        return torch.tensor(input_sample, dtype=torch.long), torch.tensor(
            output_sample, dtype=torch.long
        )


class OneLayerTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, dim_feedforward):
        super(OneLayerTransformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=1
        )
        self.fc_out = nn.Linear(d_model, output_dim)
        self.src_mask = None

    def forward(self, src):
        src = self.embedding(src)
        memory = torch.zeros(src.size(0), src.size(1), d_model).to(src.device)
        output = self.transformer_decoder(src, memory, self.src_mask)
        return self.fc_out(output)


# Generate random data
def generate_random_data(batch_size, num_tokens, input_dim, output_dim):
    inputs = torch.randint(0, input_dim, (batch_size, num_tokens))
    outputs = torch.randint(0, output_dim, (batch_size, num_tokens))
    return inputs, outputs


def train(model, dataloader):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_function(output.transpose(1, 2), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")

    # Test the model
    test_inputs, _ = generate_random_data(1, num_tokens, input_dim, output_dim)
    test_output = model(test_inputs)
    print("Test input:", test_inputs)
    print("Test output:", test_output.argmax(dim=2))


def get_dataloader(zipped_data, batch_size=32):
    # Unzip the input and output data
    input_data, output_data = zip(*zipped_data)

    # Create the custom dataset
    dataset = CustomDataset(input_data, output_data)

    # Create a DataLoader to handle batching
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "main":
    # TODO: Get training data

    model = OneLayerTransformer(input_dim, output_dim, d_model, nhead, dim_feedforward)
    train(model, get_dataloader(training_data))
