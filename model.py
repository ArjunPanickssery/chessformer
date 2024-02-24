from os import name
from re import L
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data import load_from_json

# Parameters
input_dim = 25  # Input dimension (size of one-hot token)
output_dim = 16  # Output dimension (size of output one-hot vector)
d_model = 256  # Dimension of the model
nhead = 8  # Number of heads in the multiheadattention models
dim_feedforward = 256  # Dimension of the feedforward network model
# num_tokens = 25  # Number of tokens in the sequence
batch_size = 256  # Batch size for training
# num_batches = 100  # Number of batches for training
learning_rate = 0.01  # Learning rate for the optimizer
num_epochs = 20
num_layers = 24
DATASET_NAME = "vectors_size_5"  # "squarecode_size_5"


class CustomDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        output_sample = self.output_data[idx]
        return torch.tensor(input_sample, dtype=torch.long), F.one_hot(
            torch.tensor(output_sample), output_dim
        ).to(torch.float32)


class OneLayerTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, dim_feedforward):
        super(OneLayerTransformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src):
        print(src.shape)
        src = self.embedding(src)
        print(src.shape)
        memory = torch.zeros(src.size(0), src.size(1), d_model).to(src.device)
        output = self.transformer_decoder(src, memory)
        print(output.shape)
        # Use the state of the last token for classification
        last_token_state = output[:, -1, :]
        return self.fc_out(last_token_state)


def get_dataloader(zipped_data_path, batch_size=32):
    """Expects JSON-file with list of tuples of input and output data,
    e.g. [([1, 2, 3, 4, 5, ...], 1), ([6, 7, 8, 9, 10, ...], 2), ...]"""

    zipped_data = load_from_json(zipped_data_path)

    print(f"Loading {len(zipped_data)} data-points from {zipped_data_path}")

    # Unzip the input and output data
    input_data, output_data = zip(*zipped_data[:3])

    # Create the custom dataset
    dataset = CustomDataset(input_data, output_data)

    # Create a DataLoader to handle batching
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(model, dataloader, model_name) -> None:
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            print("\n\n---")
            print(inputs)
            optimizer.zero_grad()
            output = model(inputs)
            # print(output.dtype, labels.dtype)
            # print(output.shape, labels.shape)
            print("---")
            print(output)
            print(torch.max(output, 1)[1])
            print(labels)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")

    # Save the trained model
    torch.save(model.state_dict(), f"models/{model_name}.pt")
    print(f"Model saved to models/{model_name}.pt")


def evaluate(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients for evaluation
        for inputs, labels in dataloader:
            outputs = model(inputs)
            print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            _, actual = torch.max(labels.data, 1)
            print(predicted)
            print(actual)
            correct += (predicted == actual).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the dataset: {accuracy:.2f}%")


def load_model(model_name):
    # Initialize the model
    model = OneLayerTransformer(input_dim, output_dim, d_model, nhead, dim_feedforward)
    # Load the saved state_dict
    model.load_state_dict(torch.load(f"models/{model_name}.pt"))
    model.eval()  # Set the model to evaluation mode
    return model


if __name__ == "__main__":
    model = OneLayerTransformer(input_dim, output_dim, d_model, nhead, dim_feedforward)
    dataloader = get_dataloader(
        f"training_data/{DATASET_NAME}.json", batch_size=batch_size
    )
    train(model, dataloader, model_name="24_layers_huge_3_examples_100_epochs")
    evaluate(model, dataloader)
