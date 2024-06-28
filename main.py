import numpy as np
from torch import from_numpy
from torch.jit import load
from torch.utils.data import Dataset, DataLoader
import sys

class Data(Dataset):
    def __init__(self, filename, delimiter=','):
        data = np.genfromtxt(filename, delimiter=delimiter)
        self.y = from_numpy(data[1:, 0]).float().reshape(-1, 1)
        self.x = from_numpy(data[1:, 1:]).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        y = (y + 1) * .5
        return x, y
    
def evaluate(filename):
    model = load('classifier.pt')
    model.eval()
    loader = DataLoader(Data(filename), batch_size=100)
    acc, counter = 0, 0
    for x, y in loader:
        y_pred = model(x)
        acc = (counter / (counter + 1) * acc + 1 / (counter + 1) * (y_pred.round() == y.round()).float().mean()).item()
        counter += 1
    return acc

if __name__ == "__main__":
    filename = 'training.csv'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    acc = evaluate(filename)
    print(acc)