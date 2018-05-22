import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

BATCH_SIZE = 1
IMAGE_SIZE = 28 * 28
LEARNRATE = 0.005
EPOCHS = 10
FIRST_HIDDEN_LAYER_SIZE = 100
SECOND_HIDDEN_LAYER_SIZE = 50


def main():
    ## Define our MNIST Datasets (Images and Labels) for training and testing
    train_dataset = datasets.FashionMNIST(root='./data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.FashionMNIST(root='./data',
                                  train=False,
                                  transform=transforms.ToTensor())

    # Define the indices
    indices = list(range(len(train_dataset)))  # start with all the indices in training set
    split = int(len(train_dataset)*0.2)  # define the split size
    # Define your batch_size
    batch_size = 1

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

    model = FirstNet(image_size=IMAGE_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNRATE)
    train(train_loader,validation_loader,model,optimizer,test_loader)

    print "liz"





def train(train_loader,validation_loader,model, optimizer,test_loader):
    for i in range(EPOCHS):
        print "epoch" +str(i)
        model.train()
        for data, labels in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output,labels)
            loss.backward()
            optimizer.step()
        validation(model,validation_loader)

def validation(model, validation_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in validation_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(validation_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validation_loader),
        100. * correct / len(validation_loader)))


class FirstNet(nn.Module):
    FIRST_HIDDEN_LAYER_SIZE = 100
    SECOND_HIDDEN_LAYER_SIZE = 50

    def __init__(self, image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, FIRST_HIDDEN_LAYER_SIZE)
        self.fc1 = nn.Linear(FIRST_HIDDEN_LAYER_SIZE, SECOND_HIDDEN_LAYER_SIZE)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)



if __name__ == "__main__":
    main()
