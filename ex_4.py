import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

BATCH_SIZE = 1
IMAGESIZE = 28 * 28
LEARNRATE = 0.005
EPOCHS = 10
FIRST_HIDDEN_LAYER_SIZE = 100
SECOND_HIDDEN_LAYER_SIZE = 50





    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST('./data', train=True, download=True,
    #                           transform=transform), batch_size=BATCHSIZE, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST('./data', train=False, transform=transform),
    #     batch_size=BATCHSIZE, shuffle=True)
    #
    # ## define our indices -- our dataset has 9 elements and we want a 8:4 split
    # num_train = len(train_loader.dataset.train_data)
    # indices = list(range(num_train))
    # split = int(num_train*0.2)
    #
    # # Random, non-contiguous split
    # validation_idx = np.random.choice(indices, size=split, replace=False)
    # train_idx = list(set(indices) - set(validation_idx))
    #
    # # Contiguous split
    # # train_idx, validation_idx = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_idx)
    # validation_sampler = SubsetRandomSampler(validation_idx)
    #
    # #train_loader_splitted = torch.utils.data.DataLoader(train_loader.dataset.train_data,
    #  #                                          batch_size=BATCHSIZE, sampler=train_sampler)
    #
    # #validation_loader = torch.utils.data.DataLoader(train_loader.dataset.train_data,
    # #                                                batch_size=BATCHSIZE, sampler=validation_sampler)
    #
    # #train_loader = train_loader_splitted
    # #print len(validation_loader)
    #
    # #print len(train_loader)
def main():
    ## Define our MNIST Datasets (Images and Labels) for training and testing
    train_dataset = datasets.FashionMNIST(root='./data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.FashionMNIST(root='./data',
                                  train=False,
                                  transform=transforms.ToTensor())

    ## We need to further split our training dataset into training and validation sets.

    # Define the indices
    indices = list(range(len(train_dataset)))  # start with all the indices in training set
    #split = int(len(train_dataset)*0.2)  # define the split size
    split = 12000  # define the split size


    # Define your batch_size
    batch_size = 1

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # Contiguous split
    # train_idx, validation_idx = indices[split:], indices[:split]

    # define our samplers -- we use a SubsetRandomSampler because it will return
    # a random subset of the split defined by the given indices without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    # Create the train_loader -- use your real batch_size which you
    # I hope have defined somewhere above
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, sampler=train_sampler)

    # You can use your above batch_size or just set it to 1 here.  Your validation
    # operations shouldn't be computationally intensive or require batching.
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=1, sampler=validation_sampler)

    # You can use your above batch_size or just set it to 1 here.  Your test set
    # operations shouldn't be computationally intensive or require batching.  We
    # also turn off shuffling, although that shouldn't affect your test set operations
    # either
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    model = FirstNet(image_size=IMAGESIZE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNRATE)
    train(train_loader,validation_loader,model,optimizer,test_loader)

    print "liz"


    #
    # def train():
    #     self.model.train()
    #     for data, labels in self.train_loader:
    #         self.optimizer.zero_grad()
    #         output = self.model(data)
    #         # negative log likelihood loss
    #         loss = F.nll_loss(output, labels)
    #         # calculate gradients
    #         loss.backward()
    #         # update parameters
    #         self.optimizer.step()




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

    # def train(self):
    #     self.model.train()
    #     for data, labels in self.train_loader:
    #         self.optimizer.zero_grad()
    #         output = self.model(data)
    #         # negative log likelihood loss
    #         loss = F.nll_loss(output, labels)
    #         # calculate gradients
    #         loss.backward()
    #         # update parameters
    #         self.optimizer.step()


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
        #x = F.dropout(x)

        x = self.fc1(x)
        # x = sigmoid(x)
        return F.log_softmax(x, dim=1)



if __name__ == "__main__":
    main()
