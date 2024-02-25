import torch
from torch.utils.data import Dataset, DataLoader
import json
import torch.nn as nn


class SequencesDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, label = self.data[idx]
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim * 25, num_classes)

        self.test_fc = nn.Linear(800, 21)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x.view(-1, 25 * EMBED_DIM))

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


# Parameters
INPUT_DIM = 4
EMBED_DIM = 32
NUM_HEADS = 2
HIDDEN_DIM = 128
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

for epoch in range(EPOCHS):
    # Training step
    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Every 100 epochs, evaluate the model
    if (epoch + 1) % 10 == 0:
        accuracy = evaluate_model(model, dataloader, device)
        print(f'Epoch {epoch + 1}: Accuracy = {accuracy * 100:.2f}%')

    # Optionally, print training loss or other information every epoch
    if epoch % 1 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
