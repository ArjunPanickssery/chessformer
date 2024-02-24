import json
from math import e 
import torch as t
from tqdm import tqdm
from random import shuffle

def get_data():
    with open('training_data/vectors_size_5.json', 'r') as f:
        data = json.load(f)

    new_data = []
    for inputs, label in data:
        # new inputs are indices of numbers 1, 2 and 3 in the input list
        new_inputs = [inputs.index(1), inputs.index(2), inputs.index(3)]
        new_data.append((new_inputs, label))
    return new_data


class Attn(t.nn.Module):
    """
    attn with no masking
    """

    def __init__(self, embed_dim, num_heads):
        super(Attn, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = t.nn.Linear(embed_dim, self.head_dim * num_heads)
        self.k = t.nn.Linear(embed_dim, self.head_dim * num_heads)
        self.v = t.nn.Linear(embed_dim, self.head_dim * num_heads)
        self.out_map = t.nn.Linear(self.head_dim * num_heads, embed_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, embed_dim)
        """
        batch, seq_len, embed_dim = x.size()
        q = self.q(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v(x).view(batch, seq_len, self.num_heads, self.head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(batch, seq_len, embed_dim)
        out = self.out_map(out)
        return out


class Block(t.nn.Module):
    """
    attn + norm + mlp + norm
    """

    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.attn = Attn(embed_dim, num_heads)
        self.norm1 = t.nn.LayerNorm(embed_dim)
        self.mlp = t.nn.Sequential(
            t.nn.Linear(embed_dim, embed_dim * 4),
            t.nn.GELU(),
            t.nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = t.nn.LayerNorm(embed_dim)

    def forward(self, x):
        out = self.attn(x)
        out = self.norm1(out + x)
        out = self.mlp(out)
        out = self.norm2(out + x)
        return out


class Transformer(t.nn.Module):
    """
    transformer takes in sequences of length 3 with a vocab size of 25, and predicts a label from 0 to 15 (16 options)
    """

    def __init__(self, embed_dim, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embed = t.nn.Embedding(25, embed_dim)
        self.blocks = t.nn.ModuleList([Block(embed_dim, num_heads) for _ in range(num_layers)])
        self.out = t.nn.Linear(embed_dim, 16)

    def forward(self, x):
        out = self.embed(x)
        for block in self.blocks:
            out = block(out)
        out = self.out(out.mean(dim=1))
        return out
    
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
    print(len(data), "data points")
    model = Transformer(embed_dim=32, num_heads=4, num_layers=6)
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=0.001)
    n_prints = min(20, len(data)//2)
    print_every = len(data) // n_prints
    avg_loss = 0
    n = 0
    for epoch in range(10):
        model.train()
        shuffle(data)
        for idx, (inputs, label) in tqdm(enumerate(data)):
            inputs = t.tensor(inputs).unsqueeze(0)
            label = t.tensor(label).unsqueeze(0)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, label)
            avg_loss += loss.item()
            n += 1
            loss.backward()
            optimizer.step()
            if idx % print_every == 0:
                print(f'Epoch: {epoch}, Loss: {avg_loss / n}')
                avg_loss = 0
                n = 0
        eval_accuracy(model)

    t.save(model.state_dict(), 'nina_model.pth')


if __name__ == '__main__':
    train()