import torch
import torch.nn as nn
import math
import json
from torch.utils.data import Dataset, DataLoader


class SequencesDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, label = self.data[idx]
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, num_classes, max_seq_length=25):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim * max_seq_length, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc(x.view(-1, 25 * EMBED_DIM))  # Make sure to adjust this according to your sequence length
        return x


def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    correct, total = 0, 0
    with torch.no_grad():  # No need to track gradients during evaluation
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    model.train()  # Set the model back to training mode
    return accuracy


# Your parameters remain unchanged
INPUT_DIM = 4
EMBED_DIM = 32
NUM_HEADS = 2
HIDDEN_DIM = 1024
NUM_CLASSES = 17
BATCH_SIZE = 256
EPOCHS = 20000

# Load Dataset
dataset = SequencesDataset('training_data/vectors_size_5.json')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize Model
model = TransformerClassifier(INPUT_DIM, EMBED_DIM, NUM_HEADS, HIDDEN_DIM, NUM_CLASSES)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the appropriate device


def print_num_params(model):
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params}')


print_num_params(model)

for epoch in range(EPOCHS):
    # Training step
    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation and logging
    if (epoch + 1) % 10 == 0:
        accuracy = evaluate_model(model, dataloader, device)
        print(f'Epoch {epoch + 1}: Accuracy = {accuracy * 100:.2f}%')
    if epoch % 1 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
