"""Solution for https://www.kaggle.com/competitions/spaceship-titanic/overview"""
from typing import Iterable, Callable

import torch

from torch import Tensor

PLANETS = ("Europa", "Earth", "Mars")
BOOLS = ("False", "True")
DESTINATIONS = ("55 Cancri e", "PSO J318.5-22", "TRAPPIST-1e")
DECKS = ("A", "B", "C", "D", "E", "F", "G", "T")
SIDES = ("P", "S")


def scaled_float(value: str, minval: float, maxval: float) -> tuple[float]:
    """Converts a str to float (-1 if empty string), and normalizes between 0&1"""
    if value == "":
        fvalue = -1.0
    else:
        fvalue = float(value)
    return ((fvalue - minval) / (maxval - minval),)


def parse_cabin(field: str) -> Iterable[bool | float]:
    """parses the cabin field of a passenger"""
    if field == "":
        deck = num = side = ""
    else:
        deck, num, side = field.split("/")
    deckres = one_hot(deck, DECKS)
    numres = scaled_float(num, -1, 1894)
    sideres = one_hot(side, SIDES)
    return (*deckres, *numres, *sideres)


def one_hot(value: str, values: Iterable[str]) -> list[bool]:
    """Converts an "enum" into a "one-hot tensor" """
    return [v == value for v in values]


TITANIC_TRANSFORMS: dict[str, Callable[[str], Iterable[float | bool]]] = {
    'Cabin':        parse_cabin,

    'HomePlanet':   lambda x: one_hot(x,       values=PLANETS),
    'CryoSleep':    lambda x: one_hot(x,       values=BOOLS),
    'Destination':  lambda x: one_hot(x,       values=DESTINATIONS),
    'VIP':          lambda x: one_hot(x,       values=BOOLS),

    'Age':          lambda x: scaled_float(x,  minval=-1.0, maxval=79.0),
    'RoomService':  lambda x: scaled_float(x,  minval=-1.0, maxval=14327.0),
    'FoodCourt':    lambda x: scaled_float(x,  minval=-1.0, maxval=29813.0),
    'ShoppingMall': lambda x: scaled_float(x,  minval=-1.0, maxval=23492.0),
    'Spa':          lambda x: scaled_float(x,  minval=-1.0, maxval=22408.0),
    'VRDeck':       lambda x: scaled_float(x,  minval=-1.0, maxval=24133.0),
}

TARGET = "Transported"


class TitanicDataset(torch.utils.data.Dataset):
    """Dataset to be used by dataloader"""

    def __init__(self, lines: list[str], labels: list[str], train=True):

        self.data: list[Tensor] = []
        self.target: list[Tensor] = []
        self.labels = labels
        self.train = train

        for line in lines:
            line_data = line.strip().split(",")
            values: list[float | bool] = []

            for value, label in zip(line_data, labels):
                if label not in TITANIC_TRANSFORMS:
                    continue
                values.extend(TITANIC_TRANSFORMS[label](value))

            self.data.append(Tensor(values))
            if train:
                self.target.append(Tensor(one_hot(line_data[labels.index(TARGET)],
                    BOOLS)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor] | Tensor:
        if self.train:
            return self.data[idx], self.target[idx]
        return self.data[idx]


def read_train_test() -> tuple[TitanicDataset, TitanicDataset]:
    """split dataset 90/10 for the minor amusement of seeing it get better"""
    with open("spaceship_titanic/train.csv", encoding="utf-8") as file:
        labels = file.readline().strip().split(",")
        lines = file.readlines()

    training_data = TitanicDataset(lines[: -len(lines) // 10], labels)
    test_data = TitanicDataset(lines[-len(lines) // 10 :], labels)
    return training_data, test_data


def read_train() -> TitanicDataset:
    """Read the whole dataset for final run"""
    with open("spaceship_titanic/train.csv", encoding="utf-8") as file:
        labels = file.readline().strip().split(",")
        lines = file.readlines()
    return TitanicDataset(lines, labels)


class TitanicNetwork(torch.nn.Module):
    """my little babby neural network"""

    def __init__(self, start=27, width=512):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(start, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, 2),
        )

    def forward(self, parameter: Tensor):
        """what even is x"""
        return self.linear_relu_stack(parameter)


def train_loop(dataloader, model, loss_fn, optimizer, print_output=False):
    """lift! lift! lift!"""
    loss = None
    # size = len(dataloader.dataset)
    for batch, (tensors, target) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(tensors)
        loss = loss_fn(pred, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def test_loop(dataloader, model, loss_fn) -> float:
    """test stuff"""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0.0, 0.0

    with torch.no_grad():
        for tensors, result in dataloader:
            pred = model(tensors)
            test_loss += loss_fn(pred, result).item()
            correct += (pred.argmax(1) == result.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return correct

def submit(model: TitanicNetwork) -> None:
    """write results of running model on test.csv to submission.csv"""
    with open('spaceship_titanic/test.csv', encoding='utf-8') as file:
        labels = file.readline().strip().split(',')
        lines = file.readlines()
    data = TitanicDataset(lines=lines, labels=labels, train=False)
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size = len(data),
        shuffle = False)
    res = 'poop'
    for poop in dataloader:
        print('poop', poop)
        pred = model(poop)
        print(pred[:10])
        res = pred.argmax(1)

    with open('spaceship_titanic/submission.csv', 'w', encoding='utf-8') as file:
        file.write('PassengerId,Transported\n')
        for passenger,survive in zip(lines, res):
            file.write(f'{passenger.split(",")[0]},{bool(survive)}\n')


def main(epochs=1000, learning_rate=1e-3):
    """do the training"""
    batch_size = 64
    training_data = read_train()
    train_dataloader = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )

    model = TitanicNetwork()
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    accuracy = 0

    print(learning_rate)
    for epoch in range(epochs//10*2+1):
        loss = train_loop(
            train_dataloader, model, loss_fn, optimizer, print_output=epoch % 50 == 0
        )
        if epoch%10 == 0:
            print(loss)
    learning_rate = 1e-4
    print(learning_rate)
    for epoch in range(epochs//10*3+1):
        loss = train_loop(
            train_dataloader, model, loss_fn, optimizer, print_output=epoch % 50 == 0
        )
        if epoch%10 == 0:
            print(loss)
    learning_rate = 1e-5
    print(learning_rate)
    for epoch in range(epochs//10*5+1):
        loss = train_loop(
            train_dataloader, model, loss_fn, optimizer, print_output=epoch % 50 == 0
        )
        if epoch%10 == 0:
            print(loss)

    print("quack")
    submit(model)


def train_main(epochs=1000, learning_rate=1e-3):
    """do the training"""
    batch_size = 64
    training_data, test_data = read_train_test()
    train_dataloader = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True
    )

    model = TitanicNetwork()
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    accuracy = 0

    for epoch in range(epochs+1):
        train_loop(
            train_dataloader, model, loss_fn, optimizer, print_output=epoch % 100 == 0
        )
        if epoch%10 == 0:
            accuracy = test_loop(test_dataloader, model, loss_fn)
            if accuracy > 0.7:
                learning_rate=1e-4

    print("quack")
    submit(model)


if __name__ == "__main__":
    main()
