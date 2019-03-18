import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from .tracker import Tracker

import random
import scipy.stats as st

import matplotlib.pyplot as plt

class WeightedModel:

    device = torch.device('cuda')
    data_dtype = torch.float32
    label_dtype = torch.int64

    def __init__(self, component, model, weighted_sampler, train_data, val_data):
        self.component = component
        self.model = model
        self.weighted_sampler = weighted_sampler
        self.train_data = train_data
        self.val_data = val_data
        self.tracker = Tracker()
        self.mu = 0
        self.points = np.array([])
        self.iter = 1

    def reset(self, checkpoint):
        self.model.load_state_dict(torch.load(checkpoint))

    def eval_conf(self, BATCH=64):
        eval_index = self.weighted_sampler.sample_eval_batch(BATCH)

        valid_index = random.sample(range(self.val_data.shape[0]), BATCH)
        batch = self.val_data[valid_index]
        X = torch.from_numpy(batch[:, :-1]).to(device=self.device, dtype=self.data_dtype)
        y = torch.from_numpy(batch[:, -1]).to(device=self.device, dtype=self.label_dtype)
        scores = self.model(X)
        loss = torch.nn.functional.cross_entropy(scores, y)
        print('in eval, validation loss is ', loss.item())
        self.mu = (self.iter * self.mu + loss.item()) / (self.iter + 1)
        self.iter += 1

        # weights = []
        losses = []
        for i in range(self.component):
            batch = self.train_data[eval_index[i]]
            X = torch.from_numpy(batch[:, :-2]).to(device=self.device, dtype=self.data_dtype)
            y = torch.from_numpy(batch[:, -2]).to(device=self.device, dtype=self.label_dtype)
            scores = self.model(X)
            loss = torch.nn.functional.cross_entropy(scores, y)
            self.points = np.append(self.points, loss.item())

            # print('In component ', i, ' loss is ', loss.item())
            # weights.append(1 - np.fabs(min(st_norm.cdf(loss.item()), st_norm.cdf(2 * mu - loss.item())) - std_p))
            losses.append(loss.item())
            self.model.zero_grad()

        std = (((self.points - self.mu) ** 2).sum() / self.points.shape[0]) ** 0.5
        st_norm = st.norm(self.mu, std)

        weights = []
        for i in range(self.component):
            weights.append(min(st_norm.cdf(losses[i]), st_norm.cdf(2 * self.mu - losses[i])))

        weights = np.array(weights) / np.array(weights).sum()
        print(weights)

        self.weighted_sampler.set_weight(weights)

    def eval_shifted_conf(self, BATCH=64):
        eval_index = self.weighted_sampler.sample_eval_batch(BATCH)

        valid_index = random.sample(range(self.val_data.shape[0]), BATCH)
        batch = self.val_data[valid_index]
        X = torch.from_numpy(batch[:, :-1]).to(device=self.device, dtype=self.data_dtype)
        y = torch.from_numpy(batch[:, -1]).to(device=self.device, dtype=self.label_dtype)
        scores = self.model(X)
        loss = torch.nn.functional.cross_entropy(scores, y)
        print('in eval, validation loss is ', loss.item())
        self.mu = (self.iter * self.mu + loss.item()) / (self.iter + 1)
        self.iter += 1

        #weights = []
        losses = []
        for i in range(self.component):
            batch = self.train_data[eval_index[i]]
            X = torch.from_numpy(batch[:, :-2]).to(device=self.device, dtype=self.data_dtype)
            y = torch.from_numpy(batch[:, -2]).to(device=self.device, dtype=self.label_dtype)
            scores = self.model(X)
            loss = torch.nn.functional.cross_entropy(scores, y)
            self.points = np.append(self.points, loss.item())


            #print('In component ', i, ' loss is ', loss.item())
            #weights.append(1 - np.fabs(min(st_norm.cdf(loss.item()), st_norm.cdf(2 * mu - loss.item())) - std_p))
            losses.append(loss.item())
            self.model.zero_grad()

        std = (((self.points - self.mu)**2).sum() / self.points.shape[0])**0.5
        st_norm = st.norm(self.mu, std)
        std_p = st_norm.cdf(self.mu - 1 * std)

        weights = []
        for i in range(self.component):
            weights.append(1 - np.fabs(min(st_norm.cdf(losses[i]), st_norm.cdf(2 * self.mu - losses[i])) - std_p))

        weights = np.array(weights) / np.array(weights).sum()
        print(weights)

        self.weighted_sampler.set_weight(weights)

    def eval_grad(self, BATCH=64):
        eval_index = self.weighted_sampler.sample_eval_batch(BATCH)

        eval_grads = []

        self.model.train()

        for i in range(self.component):
            batch = self.train_data[eval_index[i]]
            X = torch.from_numpy(batch[:, :-2]).to(device=self.device, dtype=self.data_dtype)
            y = torch.from_numpy(batch[:, -2]).to(device=self.device, dtype=self.label_dtype)
            scores = self.model(X)
            loss = torch.nn.functional.cross_entropy(scores, y)
            loss.backward()
            grads = []
            for w in self.model.parameters():
                grads += list(w.grad.cpu().detach().numpy().flatten())
            eval_grads.append(grads)
            self.model.zero_grad()

        valid_index = random.sample(range(self.val_data.shape[0]), BATCH)
        batch = self.val_data[valid_index]
        X = torch.from_numpy(batch[:, :-1]).to(device=self.device, dtype=self.data_dtype)
        y = torch.from_numpy(batch[:, -1]).to(device=self.device, dtype=self.label_dtype)
        scores = self.model(X)
        loss = torch.nn.functional.cross_entropy(scores, y)
        loss.backward()
        valid_grads = []
        for w in self.model.parameters():
            valid_grads += list(w.grad.cpu().detach().numpy().flatten())
        self.model.zero_grad()

        eval_matrix = np.array(eval_grads)
        valid_matrix = np.array(valid_grads).reshape(1, -1) * np.ones_like(eval_matrix)

        eps = 1e-8

        cosine_similarity = (eval_matrix * valid_matrix).sum(axis=1) / (((eval_matrix * eval_matrix).sum(axis=1) *
                                                                         (valid_matrix * valid_matrix).sum(
                                                                             axis=1)) ** 0.5 + eps)
        print(cosine_similarity)
        self.weighted_sampler.adjust_weight(0.1 * cosine_similarity)

    def eval_valid_loss(self, optimizer, lr=0.1):
        eval_index = self.weighted_sampler.sample_eval_batch(BATCH)
        prev_state = self.model.state_dict()

        self.model.train()

        with torch.no_grad:
            valid_index = random.sample(range(self.val_data.shape[0]), BATCH)
            valid_batch = self.val_data[valid_index]
            X = torch.from_numpy(valid_batch[:, :-1]).to(device=self.device, dtype=self.data_dtype)
            y = torch.from_numpy(valid_batch[:, -1]).to(device=self.device, dtype=self.label_dtype)
            scores = self.model(X)
            valid_loss = torch.nn.functional.cross_entropy(scores, y).item()

        losses = []
        for i in range(self.component):
            self.model.load_state_dict(prev_state)
            batch = self.train_data[eval_index[i]]
            X = torch.from_numpy(batch[:, :-2]).to(device=self.device, dtype=self.data_dtype)
            y = torch.from_numpy(batch[:, -2]).to(device=self.device, dtype=self.label_dtype)
            scores = self.model(X)
            loss = torch.nn.functional.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad:
                X = torch.from_numpy(valid_batch[:, :-1]).to(device=self.device, dtype=self.data_dtype)
                y = torch.from_numpy(valid_batch[:, -1]).to(device=self.device, dtype=self.label_dtype)
                scores = self.model(X)
            losses.append(valid_loss - torch.nn.functional.cross_entropy(scores, y).item())

        losses = np.array(losses)
        print(losses)

        self.weighted_sampler.adjust_weight(lr * losses)

    def train(self, optimizer, size, MODE='conf', BATCH=64, MAX_EPOCH=30, USE_WEIGHT=True):
        VAL = self.val_data.shape[0]

        train_loss_history = []
        val_loss_history = []
        train_recall_history = []
        val_recall_history = []

        for e in range(MAX_EPOCH):
            self.model.train()
            epoch_train_loss = 0
            epoch_train_recall = 0

            print("training epoch " + str(e) + " starts")

            if USE_WEIGHT:
                if MODE=='conf':
                    self.eval_conf(BATCH)
                elif MODE=='shifted_conf':
                    self.eval_shifted_conf(BATCH)
                elif MODE=='grad':
                    self.eval_grad(BATCH)
                elif MODE=='valid_loss':
                    self.eval_valid_loss(BATCH)
                weighted_index = self.weighted_sampler.sample_train_data(size)
                data = self.train_data[weighted_index]
            else:
                data = self.train_data[random.sample(range(self.train_data.shape[0]), size)]

            print(data.shape)

            NUM = data.shape[0]

            loader_train = DataLoader(data, batch_size=BATCH,
                                      sampler=sampler.SubsetRandomSampler(range(NUM)))
            loader_valid = DataLoader(self.val_data, batch_size=BATCH,
                                      sampler=sampler.SubsetRandomSampler(range(VAL)))

            train_iter = 0
            for i, batch in enumerate(loader_train):
                X = (batch[:, :-2]).to(device=self.device, dtype=self.data_dtype)
                y = (batch[:, -2]).to(device=self.device, dtype=self.label_dtype)
                scores = self.model(X)
                accuracy = np.argmax(scores.cpu().detach().numpy(), axis=1) - y.cpu().detach().numpy()
                mask = np.where(accuracy == 0)
                accuracy = len(mask[0]) / X.shape[0]
                epoch_train_recall += accuracy
                loss = torch.nn.functional.cross_entropy(scores, y)
                epoch_train_loss += loss.item()
                if i % 100 == 0:
                    print("epoch is ", e, ", training loss is ", loss.item())
                    print("epoch is ", e, ", training accuracy is ", accuracy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_iter += 1

            train_loss_history.append(epoch_train_loss / train_iter)
            train_recall_history.append(epoch_train_recall / train_iter)

            epoch_val_loss = 0
            epoch_val_recall = 0

            val_iter = 0
            for i, batch in enumerate(loader_valid):
                self.model.eval()  # put model to evaluation mode
                X = (batch[:, :-1]).to(device=self.device, dtype=self.data_dtype)
                y = (batch[:, -1]).to(device=self.device, dtype=self.label_dtype)
                with torch.no_grad():
                    scores = self.model(X)
                    accuracy = np.argmax(scores.cpu().detach().numpy(), axis=1) - y.cpu().detach().numpy()
                    mask = np.where(accuracy == 0)
                    accuracy = len(mask[0]) / X.shape[0]
                    epoch_val_recall += accuracy
                    loss = torch.nn.functional.cross_entropy(scores, y)
                    epoch_val_loss += loss.item()
                    if i % 100 == 0:
                        print("epoch is ", e, ", validation loss is ", loss.item())
                        print("epoch is ", e, ", validation accuracy is ", accuracy)
                    val_iter += 1

            val_loss_history.append(epoch_val_loss / val_iter)
            val_recall_history.append(epoch_val_recall / val_iter)

        return train_loss_history, train_recall_history, val_loss_history, val_recall_history

    def train_with_conf_eval(self, optimizer, size, BATCH=64, MAX_EPOCH=30, USE_WEIGHT=True):
        VAL = self.val_data.shape[0]
        self.model.train()

        train_loss_history = []
        val_loss_history = []
        train_recall_history = []
        val_recall_history = []

        for e in range(MAX_EPOCH):
            epoch_train_loss = 0
            epoch_train_recall = 0

            print("training epoch " + str(e) + " starts")

            if USE_WEIGHT:
                self.eval_conf(BATCH)
                weighted_index = self.weighted_sampler.sample_train_data(size)
                data = self.train_data[weighted_index]
            else:
                data = self.train_data[random.sample(range(self.train_data.shape[0]), size)]

            print(data.shape)

            NUM = data.shape[0]

            loader_train = DataLoader(data, batch_size=BATCH,
                                      sampler=sampler.SubsetRandomSampler(range(NUM)))
            loader_valid = DataLoader(self.val_data, batch_size=BATCH,
                                      sampler=sampler.SubsetRandomSampler(range(VAL)))

            train_iter = 0
            for i, batch in enumerate(loader_train):
                X = (batch[:, :-2]).to(device=self.device, dtype=self.data_dtype)
                y = (batch[:, -2]).to(device=self.device, dtype=self.label_dtype)
                scores = self.model(X)
                accuracy = np.argmax(scores.cpu().detach().numpy(), axis=1) - y.cpu().detach().numpy()
                mask = np.where(accuracy == 0)
                accuracy = len(mask[0]) / X.shape[0]
                epoch_train_recall += accuracy
                loss = torch.nn.functional.cross_entropy(scores, y)
                epoch_train_loss += loss.item()
                if i % 100 == 0:
                    print("epoch is ", e, ", training loss is ", loss.item())
                    print("epoch is ", e, ", training accuracy is ", accuracy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_iter +=1

            train_loss_history.append(epoch_train_loss / train_iter)
            train_recall_history.append(epoch_train_recall / train_iter)

            epoch_val_loss = 0
            epoch_val_recall = 0

            val_iter = 0
            for i, batch in enumerate(loader_valid):
                self.model.eval()  # put model to evaluation mode
                X = (batch[:, :-1]).to(device=self.device, dtype=self.data_dtype)
                y = (batch[:, -1]).to(device=self.device, dtype=self.label_dtype)
                with torch.no_grad():
                    scores = self.model(X)
                    accuracy = np.argmax(scores.cpu().detach().numpy(), axis=1) - y.cpu().detach().numpy()
                    mask = np.where(accuracy == 0)
                    accuracy = len(mask[0]) / X.shape[0]
                    epoch_val_recall += accuracy
                    loss = torch.nn.functional.cross_entropy(scores, y)
                    epoch_val_loss += loss.item()
                    if i % 100 == 0:
                        print("epoch is ", e, ", validation loss is ", loss.item())
                        print("epoch is ", e, ", validation accuracy is ", accuracy)
                    val_iter += 1

            val_loss_history.append(epoch_val_loss / val_iter)
            val_recall_history.append(epoch_val_recall / val_iter)

        return train_loss_history, train_recall_history, val_loss_history, val_recall_history

    def train_with_grad_eval(self, optimizer, size, BATCH=64, MAX_EPOCH=30, USE_WEIGHT=True):
        VAL = self.val_data.shape[0]
        self.model.train()

        train_loss_history = []
        val_loss_history = []
        train_recall_history = []
        val_recall_history = []

        for e in range(MAX_EPOCH):
            epoch_train_loss = 0
            epoch_train_recall = 0

            print("training epoch " + str(e) + " starts")

            if USE_WEIGHT:
                self.eval_grad(BATCH)
                weighted_index = self.weighted_sampler.sample_train_data(size)
                data = self.train_data[weighted_index]
            else:
                data = self.train_data[random.sample(range(self.train_data.shape[0]), size)]

            print(data.shape)

            NUM = data.shape[0]

            loader_train = DataLoader(data, batch_size=BATCH,
                                      sampler=sampler.SubsetRandomSampler(range(NUM)))
            loader_valid = DataLoader(self.val_data, batch_size=BATCH,
                                      sampler=sampler.SubsetRandomSampler(range(VAL)))

            train_iter = 0
            for i, batch in enumerate(loader_train):
                X = (batch[:, :-2]).to(device=self.device, dtype=self.data_dtype)
                y = (batch[:, -2]).to(device=self.device, dtype=self.label_dtype)
                scores = self.model(X)
                accuracy = np.argmax(scores.cpu().detach().numpy(), axis=1) - y.cpu().detach().numpy()
                mask = np.where(accuracy == 0)
                accuracy = len(mask[0]) / X.shape[0]
                epoch_train_recall += accuracy
                loss = torch.nn.functional.cross_entropy(scores, y)
                epoch_train_loss += loss.item()
                if i % 100 == 0:
                    print("epoch is ", e, ", training loss is ", loss.item())
                    print("epoch is ", e, ", training accuracy is ", accuracy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_iter += 1

            train_loss_history.append(epoch_train_loss / train_iter)
            train_recall_history.append(epoch_train_recall / train_iter)

            epoch_val_loss = 0
            epoch_val_recall = 0

            val_iter = 0
            for i, batch in enumerate(loader_valid):
                self.model.eval()  # put model to evaluation mode
                X = (batch[:, :-1]).to(device=self.device, dtype=self.data_dtype)
                y = (batch[:, -1]).to(device=self.device, dtype=self.label_dtype)
                with torch.no_grad():
                    scores = self.model(X)
                    accuracy = np.argmax(scores.cpu().detach().numpy(), axis=1) - y.cpu().detach().numpy()
                    mask = np.where(accuracy == 0)
                    accuracy = len(mask[0]) / X.shape[0]
                    epoch_val_recall += accuracy
                    loss = torch.nn.functional.cross_entropy(scores, y)
                    epoch_val_loss += loss.item()
                    if i % 100 == 0:
                        print("epoch is ", e, ", validation loss is ", loss.item())
                        print("epoch is ", e, ", validation accuracy is ", accuracy)
                    val_iter += 1

            val_loss_history.append(epoch_val_loss / val_iter)
            val_recall_history.append(epoch_val_recall / val_iter)

        return train_loss_history, train_recall_history, val_loss_history, val_recall_history

    def weighted_loss_train(self, optimizer, BATCH=64, MAX_EPOCH=30, USE_WEIGHT=True):
        VAL = self.val_data.shape[0]
        self.model.train()

        train_loss_history = []
        val_loss_history = []
        train_recall_history = []
        val_recall_history = []

        for e in range(MAX_EPOCH):
            epoch_train_loss = 0
            epoch_train_recall = 0

            print("training epoch " + str(e) + " starts")

            if USE_WEIGHT:
                self.eval(BATCH)
                weighted_index = self.weighted_sampler.sample_train_data(size)

            print(data.shape)

            NUM = data.shape[0]

            loader_train = DataLoader(data, batch_size=BATCH,
                                      sampler=sampler.SubsetRandomSampler(range(NUM)))
            loader_valid = DataLoader(self.val_data, batch_size=BATCH,
                                      sampler=sampler.SubsetRandomSampler(range(VAL)))

            train_iter = 0
            for i, batch in enumerate(loader_train):
                X = (batch[:, :-2]).to(device=self.device, dtype=self.data_dtype)
                y = (batch[:, -2]).to(device=self.device, dtype=self.label_dtype)
                scores = self.model(X)
                accuracy = np.argmax(scores.cpu().detach().numpy(), axis=1) - y.cpu().detach().numpy()
                mask = np.where(accuracy == 0)
                accuracy = len(mask[0]) / X.shape[0]
                epoch_train_recall += accuracy
                loss = torch.nn.functional.cross_entropy(scores, y)

                epoch_train_loss += loss.item()
                if i % 100 == 0:
                    print("epoch is ", e, ", training loss is ", loss.item())
                    print("epoch is ", e, ", training accuracy is ", accuracy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_iter +=1

            train_loss_history.append(epoch_train_loss / train_iter)
            train_recall_history.append(epoch_train_recall / train_iter)

            epoch_val_loss = 0
            epoch_val_recall = 0

            val_iter = 0
            for i, batch in enumerate(loader_valid):
                self.model.eval()  # put model to evaluation mode
                X = (batch[:, :-1]).to(device=self.device, dtype=self.data_dtype)
                y = (batch[:, -1]).to(device=self.device, dtype=self.label_dtype)
                with torch.no_grad():
                    scores = self.model(X)
                    accuracy = np.argmax(scores.cpu().detach().numpy(), axis=1) - y.cpu().detach().numpy()
                    mask = np.where(accuracy == 0)
                    accuracy = len(mask[0]) / X.shape[0]
                    epoch_val_recall += accuracy
                    loss = torch.nn.functional.cross_entropy(scores, y)
                    epoch_val_loss += loss.item()
                    if i % 100 == 0:
                        print("epoch is ", e, ", validation loss is ", loss.item())
                        print("epoch is ", e, ", validation accuracy is ", accuracy)
                    val_iter += 1

            val_loss_history.append(epoch_val_loss / val_iter)
            val_recall_history.append(epoch_val_recall / val_iter)

        return train_loss_history, train_recall_history, val_loss_history, val_recall_history


