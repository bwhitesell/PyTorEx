from collections import Iterable
import torch
from torch import nn
from torch import optim

from .utils import eval_in_batches, Accuracy


class NeuralNetwork(nn.Module):

    def to_gpu(self, device=torch.device('cuda', 0)):
        self.to(device)

    def forward(self, x):
        return x

    def fit(self, train_dl, val_dl, epochs, lr=.001, criterion=nn.CrossEntropyLoss(), metric=Accuracy(),
            optimizer=optim.Adam, weight_decay=0, layer_lrs=[]):

        # check if were scheduling the learning rate
        if isinstance(lr, Iterable):
            lr_start = lr[0]
            sched = True
        else:
            lr_start = lr
            sched = False

        if layer_lrs:
            optimizer = optimizer(layer_lrs, lr=lr_start, weight_decay=weight_decay)
        else:
            optimizer = optimizer(self.parameters(), lr=lr_start, weight_decay=weight_decay)
        n_batches = len(train_dl)

        for epoch in range(epochs):  # loop over the training data
            train_loss = 0
            self.train()  # tell the layers to adopt training behaviors

            for i, data in enumerate(train_dl, 0):
                inputs, labels = data
                if sched:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr[i]

                # forward
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                train_loss += loss.item()
                # backward
                loss.backward()
                # optimize
                optimizer.step()
                self.zero_grad()

                # print statistics
                print(f'Epoch: {epoch}, batch {i}/{n_batches} loss: {loss.item()}', end='\r')

            # evaluate model performance on the validation set
            with torch.no_grad():
                self.eval()  # tell layers to adopt eval behaviors
                val_loss, mtrk = eval_in_batches(val_dl, self, criterion, metric=metric)
                print(f'Epoch: {epoch}, batch {i}/{n_batches}] Test loss: {val_loss}' \
                      + f' Train loss: {train_loss / i} | Test Metric: {mtrk}'
                      , end='\n')

        print('Finished Training')
