import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

# global params
EPOCHS = 10
FIRST_HIDDEN_LAYER_SIZE = 100
SECOND_HIDDEN_LAYER_SIZE = 50
IMAGE_SIZE = 784
LR = 0.05

# Define your batch_size
batch_size = 64


def main():
    """""
    main function.
    runs the program.
    implement of NN.

    """""
    ## Define our MNIST Datasets (Images and Labels) for training and testing
    train_dataset = datasets.CIFAR10(root='./data',
                                          train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

    test_dataset = datasets.CIFAR10(root='./data',
                                         train=False,
                                         transform=transforms.ToTensor())

    # Define the indices
    indices = list(range(len(train_dataset)))  # start with all the indices in training set
    split = int(len(train_dataset) * 0.2)  # define the split size

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # train_idx, validation_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    # Create the train_loader -- use your real batch_size which you
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=1, sampler=validation_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=LR)
    train(train_loader, validation_loader, model, optimizer, test_loader)
    write_test_pred(model, test_loader)


def write_test_pred(loader, best_model):
    """""
    write_test_pred function.
    runs the nn on the test set.
    """""
    # save test.pred
    pred_file = open("test.pred", 'w')
    real_file = open("real.pred", 'w')
    for data, target in loader:
        output = best_model(data)
        pred = output.data.max(1, keepdim=True)[1]
        pred_file.write(str(pred.item()) + "\n")
        real_file.write(str(target) + "\n")

    pred_file.close()
    real_file.close()


def train(train_loader, validation_loader, model, optimizer, test_loader):
    """""
    train function.
    trains the model and runs the nn on the train and validation loaders.
    """""
    dict_train_results = {}
    dict_val_results = {}
    for i in range(EPOCHS):
        print "epoch" + str(i)
        model.train()
        for data, labels in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
        loss = run_and_print_results(model, validation_loader, "validation set", 1)
        dict_val_results[i + 1] = loss
        loss = run_and_print_results(model, train_loader, "train set", batch_size)
        dict_train_results[i + 1] = loss

    # plot the results
    label1, = plt.plot(dict_val_results.keys(), dict_val_results.values(), "b-", label='validation loss')
    label2, = plt.plot(dict_train_results.keys(), dict_train_results.values(), "r-", label='train loss')
    plt.legend(handler_map={label1: HandlerLine2D(numpoints=4)})
    plt.show()


def run_and_print_results(model, loader, loader_type, batch_size):
    """""
    run_and_print_results function.
    apply the nn on the val set and train set.
    """""
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= (len(loader) * batch_size)
    print('\n' + loader_type + ': Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, (len(loader) * batch_size),
        100. * correct / (len(loader) * batch_size)))
    return test_loss


def write_test_pred(model, loader):
    """""
    write_test_pred function.
    runs the test set and writes the prediction to file.
    """""
    # save test.pred
    pred_file = open("test.pred", 'w')
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        pred_file.write(str(pred.item()) + "\n")
    test_loss /= (len(loader))
    print('\n Test Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, (len(loader)),
        100. * correct / (len(loader))))

    pred_file.close()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #3 - rgb (input)
        #6 - number of filters
        #5 - F - size of filter
        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.batchN = nn.BatchNorm1d(120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.batchN(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    main()
