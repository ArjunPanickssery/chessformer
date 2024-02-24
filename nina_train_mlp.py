import json
import torch as t
from tqdm import tqdm
from random import shuffle

def get_data():
    with open('training_data/vectors_size_5.json', 'r') as f:
        data = json.load(f)

    new_data = []
    for inputs, label in data:
        # new inputs are indices of numbers 1, 2 and 3 in the input list - can be from 0 to 24
        new_inputs = [inputs.index(1), inputs.index(2), inputs.index(3)]
        new_data.append((new_inputs, label))
    return new_data


def print_num_params(model):
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params}')


class MLPModel(t.nn.Module):
    def __init__(self, embed_dim, hidden_size, n_layers, n_classes):
        super(MLPModel, self).__init__()
        self.embedding = t.nn.Embedding(25, embed_dim)
        self.linear1 = t.nn.Linear(embed_dim * 3, hidden_size)
        self.relu = t.nn.ReLU()
        self.layers = t.nn.ModuleList([t.nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)])
        self.out = t.nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """
        x: (batch, 3)
        """
        x = self.embedding(x)  # (batch, 3, embed_dim)
        x = x.view(x.size(0), -1)  # (batch, 3 * embed_dim)
        x = self.linear1(x)  # (batch, hidden_size)
        x = self.relu(x)
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x) # (batch, hidden_size)
        x = self.out(x) # (batch, n_classes)
        return x

def eval_accuracy(model):
    data = get_data()
    model.eval()
    correct = 0
    for inputs, label in data:
        inputs = t.tensor(inputs).unsqueeze(0)
        label = t.tensor(label).unsqueeze(0)
        output = model(inputs)
        pred = output.argmax(dim=1)
        if pred == label:
            correct += 1
    print(f'Accuracy: {correct / len(data)}')


def train():
    data = get_data()
    shuffle(data)
    model = MLPModel(embed_dim=64, hidden_size=128, n_layers=2, n_classes=16)
    print_num_params(model)
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=0.0003)
    n_prints = min(5, len(data)//2)
    for epoch in range(30):
        shuffle(data)
        total_loss = 0
        pbar = tqdm(enumerate(data), total=len(data))
        for i, (inputs, label) in pbar:
            inputs = t.tensor(inputs).unsqueeze(0)
            label = t.tensor(label).unsqueeze(0)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % (len(data) // n_prints) == 0:
                pbar.write(f'Epoch: {epoch}, Loss: {total_loss / (i + 1)}')
        eval_accuracy(model)


if __name__ == '__main__':
    train()