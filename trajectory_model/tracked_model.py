import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from .tracker import Tracker

class Flatten(nn.Module):
    def forward(self, x):
        N = x.shape[0]
        return x.view(N,-1)

class TrackedModel:
    model = None
    tracker = None
    data = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    def __init__(self, model, data, val_data):
        self.model = model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.tracker = Tracker()
        self.data = data
        self.val_data = val_data
        self.criterion = nn.CrossEntropyLoss()

    def load_model(self, param_file):
        self.model.load_state_dict(torch.load(param_file))

    def reset_data(self, data):
        self.data = data

    def train(self, optimizer, BATCH=64, MAX_EPOCH=80, MILESTONES=[40,60], GAMMA=0.1, MAX_ACCURACY=0.85, USE_VAL=False):
        NUM = self.data.shape[0]
        VAL = self.val_data.shape[0]

        loader_train = DataLoader(self.data, batch_size=BATCH,
                                  shuffle=True)
        loader_valid = DataLoader(self.val_data, batch_size=BATCH,
                                  shuffle=True)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)

        train_history = None
        val_history = None
        class_history = None

        train_accuracy = []
        val_accuracy = []

        for e in range(MAX_EPOCH):
            scheduler.step()
            self.model.train()
            epoch_train_history = None
            epoch_class_history = None

            epoch_train_correct = 0

            print("training epoch " + str(e) + " starts")
            for param_group in optimizer.param_groups:
                print("learning rate is ", param_group['lr'])
            for i, batch in enumerate(loader_train):
                X = (batch[:, :-2]).to(device=self.device, dtype=self.dtype)
                y = (batch[:, -2]).to(device=self.device, dtype=torch.long)
                c = (batch[:, -1]).to(device=self.device, dtype=torch.long)
                optimizer.zero_grad()
                scores = self.model(X)
                # index = np.argmax(scores.cpu().detach().numpy(), axis=1) - y.cpu().detach().numpy()
                # mask = np.where(index == 0)
                # correct = len(mask[0])

                _, predicted = torch.max(scores.data, 1)
                correct = predicted.eq(y.data).cpu().sum().numpy()

                accuracy = correct / X.shape[0]
                epoch_train_correct += correct

                if epoch_train_history is None:
                    epoch_train_history = self.tracker.track(scores.detach(), y)
                    epoch_class_history = c
                else:
                    epoch_train_history = np.hstack((epoch_train_history, self.tracker.track(scores.detach(), y)))
                    epoch_class_history = np.hstack((epoch_class_history, c))
                loss = self.criterion(scores, y)
                if i % 100 == 0:
                    # print(i)
                    # print('X is ', X)
                    # print(scores)
                    print("epoch is ", e, ", training loss is ", loss.item())
                    print("epoch is ", e, ", training accuracy is ", accuracy)
                loss.backward()
                optimizer.step()

            epoch_train_history = epoch_train_history.reshape(NUM, 1)
            epoch_class_history = epoch_class_history.reshape(NUM, 1)

            epoch_train_accuracy = epoch_train_correct / NUM
            train_accuracy.append(epoch_train_accuracy)

            if train_history is None:
                train_history = epoch_train_history
                class_history = epoch_class_history
            else:
                train_history = np.hstack((train_history, epoch_train_history))
                class_history = np.hstack((class_history, epoch_class_history))

            if not USE_VAL:
                continue

            epoch_val_history = None

            epoch_val_correct = 0
            epoch_val_total = 0
            # best_test = 0

            for i, batch in enumerate(loader_valid):
                self.model.eval()  # put model to evaluation mode
                X = (batch[:, :-1]).to(device=self.device, dtype=self.dtype)
                y = (batch[:, -1]).to(device=self.device, dtype=torch.long)
                with torch.no_grad():
                    scores = self.model(X)

                    # index = np.argmax(scores.cpu().detach().numpy(), axis=1) - y.cpu().detach().numpy()
                    # mask = np.where(index == 0)
                    # correct = len(mask[0])

                    _, predicted = torch.max(scores.data, 1)
                    correct = predicted.eq(y.data).cpu().sum().numpy()

                    accuracy = correct / X.shape[0]
                    epoch_val_total += X.shape[0]
                    epoch_val_correct += correct
                    loss = self.criterion(scores, y)
                    # print("epoch is ", e, ", test loss is ", loss.item())
                    # print("epoch is ", e, ", test accuracy is ", accuracy)
                    # if accuracy > best_test:
                    #    best_test = accuracy
                    if i % 100 == 0:
                        print("epoch is ", e, ", test loss is ", loss.item())
                        print("epoch is ", e, ", test accuracy is ", accuracy)
                    if epoch_val_history is None:
                        epoch_val_history = self.tracker.track(scores.detach(), y)
                    else:
                        epoch_val_history = np.hstack((epoch_val_history, self.tracker.track(scores.detach(), y)))

            # print(epoch_train_history.shape)
            # print(epoch_val_history.shape)

            epoch_val_accuracy = epoch_val_correct / VAL
            print('Epoch Val Total is ', epoch_val_total, ' VAL is ', VAL)

            epoch_val_history = epoch_val_history.reshape(VAL, 1)
            val_accuracy.append(epoch_val_accuracy)

            if val_history is None:
                val_history = epoch_val_history
            else:
                val_history = np.hstack((val_history, epoch_val_history))

            print('')
            print('epoch train accuracy is ', epoch_train_accuracy)
            print('epoch test accuracy is ', epoch_val_accuracy)
            # print('epoch best accuracy is ', best_test)
            if epoch_val_accuracy > MAX_ACCURACY:
                break
        #print(train_history.shape)
        #print(val_history.shape)

        return train_history, val_history, class_history, train_accuracy, val_accuracy