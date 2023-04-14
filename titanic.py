# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MDaSSsFDw2jJ2FNsbz11VmRAc3xNoNWX

# Notebook for Kaggle competition, Titanic - Machine Learning from Disaster
"""

# Commented out IPython magic to ensure Python compatibility.
import torch
import numpy as np
# %pdb on

# upload csv files
#from google.colab import files
#uploaded = files.upload()

import torch.utils.data

def safe_float(x):
    if x == '':
        return -1.0
    return float(x)

TITANIC_TYPES = {
    'PassengerId': int,
    'Survived': bool,
    'Pclass': int,
    'Name': str,
    'Sex': lambda x: x=='female',
    'Age': safe_float,
    'SibSp': int,
    'Parch': int,
    'Ticket': str,
    'Fare': safe_float,
    'Cabin': str,
    'Embarked': lambda x: ['C', 'Q', 'S', ''].index(x)
}

TITANIC_TRANSFORMS = {
    'Pclass': lambda x: [x=='1', x=='2', x=='3'],
    'Sex': lambda x: [x=='female'],
    'Age': safe_float,
    'SibSp': safe_float,
    'Parch': safe_float,
    #'Ticket': str,
    'Fare': safe_float,
    #'Cabin': str,
    'Embarked': lambda x: [x=='C', x=='Q', x=='S', x=='']
}

# may be made a named tuple
TITANIC_TARGET_TRANSFORM = ('Survived', safe_float)

def read_csv_line(line):
    """Reads a line, splitting on ',' unless it's surrounded by '"'. Could
    easily be made significantly more efficient"""
    line_data = ['']
    quoted_value = False
    for c in line:
        if c == '"':
            quoted_value = not quoted_value
        elif c == ',' and not quoted_value:
            line_data.append('')
        else:
            line_data[-1] += c
    assert not quoted_value
    return line_data

import typing
def transform_line_data(labels, raw_line_data, normalize, train):
    values = []
    for key, data in zip(labels, raw_line_data):
        if key not in TITANIC_TRANSFORMS:
            continue
        value = TITANIC_TRANSFORMS[key](data)
        if type(value) == float:
            #values.append((value - normalize[key][0]) / normalize[key][1])
            valmax,valmin = normalize[key]
            # normalize between 0 and 1
            value = (value - valmin) / (valmax - valmin)
            # cap in case we hit out of bounds
            #value = min(max(0, value), 1)
            # This should use a better method
            values.append(value)
        else:
            values.extend(value)
    if train:
        target = TITANIC_TARGET_TRANSFORM[1](
            raw_line_data[labels.index(TITANIC_TARGET_TRANSFORM[0])])
        return torch.Tensor(values), torch.Tensor([target, not target])
    return torch.Tensor(values)

class TitanicDataset(torch.utils.data.Dataset):

    def __init__(self, labels: typing.List[str],
                 lines: typing.List[str],
                 transform=False,
                 normalize=[],
                 train=True):


        self.transform = transform
        self.data: list[list] = []
        self.transform_data = []
        self.labels = labels

        for line in lines:
            line_data = []
            raw_line_data = read_csv_line(line.strip())

            assert len(raw_line_data) == len(self.labels)
            if not transform:
                for key,data in zip(self.labels, raw_line_data):
                    line_data.append(TITANIC_TYPES[key](data))
                self.data.append(line_data)
            else:
                self.transform_data.append(transform_line_data(
                    self.labels, raw_line_data, normalize, train))

    def __len__(self):
        if self.transform:
            return len(self.transform_data)
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform_data[idx]
        return self.data[idx]


with open('titanic/train.csv', encoding='utf-8') as f:
    labels = f.readline().strip().split(',')
    lines = f.readlines()

# Find out mean and stdev of float values for normalizing

tdata = TitanicDataset(labels=labels, lines=lines, transform=False)
def get_mean_and_stdev(key):
    ages = [k[tdata.labels.index(key)] for k in tdata]
    mean_age = sum(ages) / len(ages)
    stdev_age = np.std(ages)
    return mean_age, stdev_age
def get_max_and_min(key):
    ages = [k[tdata.labels.index(key)] for k in tdata.data]
    return max(ages), min(ages)

normalize = {}
for k in 'Age', 'SibSp', 'Parch', 'Fare':
    #normalize[k] = get_mean_and_stdev(k)
    normalize[k] = get_max_and_min(k)
print(normalize)

print(normalize)
#training_data = TitanicDataset(lines=lines[:-len(lines)//10], labels=labels, transform=True,
#                               normalize=normalize)
training_data = TitanicDataset(lines=lines[:], labels=labels, transform=True,
                               normalize=normalize)
test_data = TitanicDataset(lines=lines[-len(lines)//10:], labels=labels, transform=True,
                               normalize=normalize)
#test_data = TitanicDataset(filename='test.csv')

train_dataloader = torch.utils.data.DataLoader(
    training_data,
    batch_size = 64,
    shuffle = False)
test_dataloader = torch.utils.data.DataLoader(
    test_data,
    batch_size = 64,
    shuffle = True)

k, r= next(iter(train_dataloader))
print(k.size())
print(r.size())
#print(k)

width = 256
import torch.nn
class TitanicNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(12, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, 2)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

#X = torch.rand(1, 12)
#model = TitanicNetwork()
#logits = model(X)
#print(logits)
#pred_probab = torch.nn.Softmax(dim=1)(logits)
#pred_probab

def train_loop(dataloader, model, loss_fn, optimizer, print_output=False):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        #Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0 and print_output:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}')

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')

model = TitanicNetwork()
learning_rate = 1e-3
#batch_size = 8
epochs = 3000
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    #print(f'epoch {t}', end=' ')
    train_loop(train_dataloader, model, loss_fn, optimizer, print_output=t%100==0)
    #test_loop(test_dataloader, model, loss_fn)

train_loop(train_dataloader, model, loss_fn, optimizer, print_output=True)
print('quack')

# Commented out IPython magic to ensure Python compatibility.
# %pdb off
def submit():
    results = []
    with open('titanic/test.csv', encoding='utf-8') as f:
        labels = f.readline().strip().split(',')
        lines = f.readlines()
    data = TitanicDataset(lines=lines, labels=labels, transform=True, normalize=normalize,
                          train=False)
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size = len(data),
        shuffle = True)
    res = 'poop'
    for poop in dataloader:
        print('poop', poop)
        pred = model(poop)
        print(pred[:10])
        res = pred.argmax(1)

    with open('submission.csv', 'w', encoding='utf-8') as f:
        f.write('PassengerId,Survived\n')
        for idx, survive in zip(range(892, 1310), res):
            f.write(f'{idx},{int(survive)}\n')
submit()
