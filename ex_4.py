import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

BATCHSIZE = 1
IMAGESIZE = 28 * 28
LEARNRATE = 0.005
EPOCHS = 30
FIRSTHIDDENLAYERSIZE = 50



def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=True, download=True,
                              transform=transform), batch_size=BATCHSIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, transform=transform),
        batch_size=BATCHSIZE, shuffle=True)

    ## define our indices -- our dataset has 9 elements and we want a 8:4 split
    num_train = len(train_loader.dataset.train_data)
    indices = list(range(num_train))
    split = int(num_train*0.2)
    #split =4

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # Contiguous split
    # train_idx, validation_idx = indices[split:], indices[:split]

    ## define our samplers -- we use a SubsetRandomSampler because it will return
    ## a random subset of the split defined by the given indices without replaf
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader1 = torch.utils.data.DataLoader(train_loader.dataset.train_data,
                                               batch_size=BATCHSIZE, sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(train_loader.dataset.train_data,
                                                    batch_size=BATCHSIZE, sampler=validation_sampler)

    print len(validation_loader)
    print len(train_loader1)
    #train_loader.
    # val_size = int(len(train_loader.dataset.train_data) * 0.2)
    # val_x = train_x[-val_size:, :]
    # val_y = train_y[-val_size:]
    # train_x = train_x[: -val_size, :]
    # train_y = train_y[: -val_size]

    print "liz"



    #my_obj = Pytorchi()
    #my_obj.sub()

class Pytorchi(object):

    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=True, download=True,
                                  transform=self.transform), batch_size=BATCHSIZE, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=self.transform),
            batch_size=BATCHSIZE, shuffle=True)
        self.model = FirstNet(image_size=IMAGESIZE)
        self.optimizer = optim.SGD(self.model.parameters(), lr=LEARNRATE)

    def sub(self):
        for epoch in range(1, EPOCHS + 1):
            self.train()
            self.test(epoch)

    def test(self, epoch_num):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        test_loss /= len(self.test_loader.dataset)
        print('\n epoch number :{} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_num, test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    def train(self):
        self.model.train()
        for data, labels in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(data)
            # negative log likelihood loss
            loss = F.nll_loss(output, labels)
            # calculate gradients
            loss.backward()
            # update parameters
            self.optimizer.step()


class FirstNet(nn.Module):
    def __init__(self, image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, FIRSTHIDDENLAYERSIZE)
        self.fc1 = nn.Linear(FIRSTHIDDENLAYERSIZE, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.sigmoid(self.fc0(x))
        #x = F.dropout(x)

        x = self.fc1(x)
        # x = sigmoid(x)
        return F.log_softmax(x, dim=1)



if __name__ == "__main__":
    main()
