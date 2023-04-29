import torch
import isaac
from torchvision import datasets, transforms
import argparse
import random
import tqdm
import numpy as np

torch.set_num_threads(2)

if __name__ == '__main__':

    #####################################################

    parser = argparse.ArgumentParser(description='ISAAC Newton MNIST example.')

    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--la', type=float, default=10., help='lambda_a')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-nil', '--num_isaac_layers', default=3, type=int, help='Number of layers (starting with the '
                                                                                'first) to which ISAAC should be '
                                                                                'applied. -nil 0 means no isaac.')
    parser.add_argument('--device', default='cpu', type=str)

    args = parser.parse_args()

    if args.lr is None:
        if args.num_isaac_layers == 0:
            args.lr = 0.1
        else:
            args.lr = 1.

    print(vars(args))

    #####################################################

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        isaac.Linear(784, 400, la=args.la) if args.num_isaac_layers >= 1 else torch.nn.Linear(784, 400),
        torch.nn.ReLU(),
        isaac.Linear(400, 400, la=args.la) if args.num_isaac_layers >= 2 else torch.nn.Linear(400, 400),
        torch.nn.ReLU(),
        isaac.Linear(400, 400, la=args.la) if args.num_isaac_layers >= 3 else torch.nn.Linear(400, 400),
        torch.nn.ReLU(),
        isaac.Linear(400, 400, la=args.la) if args.num_isaac_layers >= 4 else torch.nn.Linear(400, 400),
        torch.nn.ReLU(),
        isaac.Linear(400, 10, la=args.la) if args.num_isaac_layers >= 5 else torch.nn.Linear(400, 10),
    ).to(args.device)

    print(model)

    optim = torch.optim.SGD(params=model.parameters(), lr=args.lr)

    for epoch in tqdm.trange(args.n_epochs):

        # TRAIN
        model.train()
        train_acc = []
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            y_hat = model(x)
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_acc.append((y_hat.argmax(1) == y).float().mean().item())
        train_acc = np.mean(train_acc)

        # TEST
        model.eval()
        test_acc = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(args.device), y.to(args.device)
                y_hat = model(x)
                test_acc.append((y_hat.argmax(1) == y).float().mean().item())
        test_acc = np.mean(test_acc)

        print('{}:  Train: {:.3f},  Test: {:.3f},  Num ISAAC layers: {}'.format(
            epoch, train_acc, test_acc, args.num_isaac_layers
        ))

