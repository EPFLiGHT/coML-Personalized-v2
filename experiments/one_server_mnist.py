"""
Example from tutorial
1. replacing optimizer to SGD with lr=0.1.
2. Rank 0 node is assumed to be the server.
3. dataset are not split.

mpirun -np 2 python one_server_mnist.py

TODO: delete this file in the future.

"""
from __future__ import print_function
import argparse
import os
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

required = object()


def aggregate_gradients(model, world_size, average_models=False):
    """Average gradients of models across all processes."""
    rank = dist.get_rank()

    for ind, param in enumerate(model.parameters()):
        gather_list = [torch.zeros_like(param) for _ in range(world_size)] if rank == 0 else []
        tensor = torch.zeros_like(param) if rank == 0 else param.grad.data
        dist.gather(tensor=tensor, dst=0, gather_list=gather_list)

        if rank == 0:
            t = sum(gather_list[1:])
            t /= (world_size - 1) if average_models else 1
        else:
            t = torch.zeros_like(param)

        dist.broadcast(tensor=t, src=0)

        if rank != 0:
            a = ((param.grad.data - t).abs().sum())
            if a > 1e-7:
                print(a)
            param.grad.data = t


class CentralizedSGD(torch.optim.SGD):
    r"""Implements centralized stochastic gradient descent (optionally with momentum).
    Args:
        world_size (int): Size of the network
        model (:obj:`nn.Module`): Model which contains parameters for SGD
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        average_models (bool): Whether to average models together. (default: `True`)
    """

    def __init__(self,
                 world_size,
                 model,
                 lr=required,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False,
                 average_models=True):
        super(CentralizedSGD, self).__init__(model.parameters(),
                                             lr,
                                             momentum,
                                             dampening,
                                             weight_decay,
                                             nesterov)
        if average_models:
            self.agg_mode = 'avg'
        else:
            raise NotImplementedError(
                "Only average model is supported right now.")

        self.model = model
        # self.agg = AllReduceAggregation(world_size=world_size).agg_grad
        self.world_size = world_size

    def step(self, closure=None):
        """ Aggregates the gradients and performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        aggregate_gradients(self.model, self.world_size, self.agg_mode == 'avg')
        loss = super(CentralizedSGD, self).step(closure=closure)
        return loss


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def dist_train(args, model, device, train_loader, optimizer, epoch, **kwargs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if dist.get_rank() != 0:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        else:
            optimizer.step()


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    ###################################################################################
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    if world_size <= 1:
        raise ValueError("world_size should be >= 2.")

    dist.init_process_group('mpi', rank=world_rank, world_size=world_size)
    print("I am {rank} of {size} in {hostname}".format(
        rank=world_rank, size=world_size, hostname=socket.gethostname()))
    ###################################################################################

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = CentralizedSGD(world_size, model, lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        dist_train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
