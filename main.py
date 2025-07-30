import argparse
import copy
import logging
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import Subset, DataLoader
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import datasets, transforms


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_model(model_type, num_classes, name):
    if model_type == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif model_type == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif model_type == 'resnet50':
        model = resnet50(num_classes=num_classes)
    else:
        raise ValueError(model_type)
    logging.info('Model parameters of %s_%s: %2.2fM' % (
        name, model_type, (sum(p.numel() for p in model.parameters()) / (1024 * 1024))))
    return model


def init_optimizer(optimizer_type, model, lr, weight_decay=0., momentum=0.):
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    else:
        raise ValueError(optimizer_type)
    return optimizer


def init_scheduler(scheduler_type, optimizer, epochs):
    if scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    elif scheduler_type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.1, 0.3, 0.5] * epochs, gamma=0.3)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise ValueError(scheduler_type)
    return scheduler


def load_dataset(args):
    if args.dataset == 'cifar10':
        # Private dataset
        imgTransform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=imgTransform)
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=imgTransform)
        # Public dataset
        imgTransform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
        public_dataset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=imgTransform)

        args.num_classes = len(train_dataset.classes)
        args.nc = 3
        args.img_size = 32
    elif args.dataset == 'cifar100':
        # Private dataset
        imgTransform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])

        train_dataset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=imgTransform)
        test_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=imgTransform)
        # Public dataset
        imgTransform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
        public_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=imgTransform)

        args.num_classes = len(train_dataset.classes)
        args.nc = 3
        args.img_size = 32
    else:
        raise ValueError(args.dataset)
    return train_dataset, test_dataset, public_dataset


def partition_dataset(args, train_dataset):
    y_train = np.array(train_dataset.targets)
    if args.partition == 'iid':  # iid distribution
        idxs = np.random.permutation(len(y_train))
        idx_batch = np.array_split(idxs, args.K)
        user_dataidx_map = {k: idx_batch[k] for k in range(args.K)}
    elif args.partition == 'dirichlet':  # non-iid with Dirichlet distribution
        idx_batch = [[] for _ in range(args.K)]
        for c in range(args.num_classes):
            idx_c = np.where(y_train == c)[0]
            np.random.shuffle(idx_c)
            proportions = np.random.dirichlet(np.repeat(args.alpha, args.K))
            proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            idx_batch = [idx_c_k + idx.tolist() for idx_c_k, idx in zip(idx_batch, np.split(idx_c, proportions))]
        total = 0
        user_dataidx_map = {}
        for k in range(args.K):
            np.random.shuffle(idx_batch[k])
            user_dataidx_map[k] = idx_batch[k]
            total += len(idx_batch[k])
        assert total == len(y_train)
    else:
        raise ValueError(args.partition)
    local_datasets = {}
    for k, dataidx in user_dataidx_map.items():
        local_datasets[k] = Subset(train_dataset, indices=dataidx)

    user_cls_counts = {}
    for k, dataidx in user_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        user_cls_counts[k] = tmp
    del train_dataset, y_train
    assert len(local_datasets) == len(user_cls_counts) == args.K
    return local_datasets, user_cls_counts


class Server:
    def __init__(self, args, id, model_type, test_dataset=None, public_dataset=None):
        self.args = args
        self.id = id
        self.name = 'server'

        self.public_dataset = public_dataset
        self.test_dataset = test_dataset

        self.device = args.device
        self.batch_size = args.batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = args.E
        self.lr = args.lr
        self.model_type = model_type
        # self.num_classes = args.num_classes
        self.model = init_model(self.model_type, args.num_classes, self.name)
        self.optimizer_type = args.optimizer
        self.scheduler_type = args.scheduler
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.temperature = args.temperature

    def select_active_clients(self, K, C):
        m = max(math.ceil(C * K), 1)
        selected_clients = sorted(np.random.choice(range(1, K + 1), m, replace=False))
        return selected_clients

    def merge(self, local_parameters, local_weights=None):  # for FedAvg
        ensemble_parameters = copy.deepcopy(local_parameters[0])
        for n in ensemble_parameters.keys():
            ensemble_parameters[n] = 0.
            for k in range(len(local_parameters)):
                if local_weights is not None:
                    ensemble_parameters[n] += local_weights[k] * local_parameters[k][n]
                else:
                    ensemble_parameters[n] += (1 / len(local_parameters)) * local_parameters[k][n]
        return ensemble_parameters

    def test(self):
        self.model.to(self.device)

        test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size)

        self.model.eval()
        test_loss = 0
        test_acc = 0
        for images, labels in test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.no_grad():
                logits = self.model(images)
            loss = self.criterion(logits, labels)

            test_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            test_acc += accuracy_score(y_pred=preds.detach().cpu(), y_true=labels.detach().cpu())
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

        self.model.cpu()
        return test_acc, test_acc

    def active_data_sampling(self, N_query):
        self.model.to(self.device)

        public_loader = DataLoader(self.public_dataset, shuffle=False, batch_size=self.batch_size)

        # Logits
        self.model.eval()
        public_logits = None
        for images, _ in public_loader:
            images = images.to(self.device)
            with torch.no_grad():
                logits = self.model(images)
            #################################
            logits = torch.softmax(logits / self.temperature, dim=-1)
            #################################
            if public_logits is None:
                public_logits = logits.detach().cpu()
            else:
                public_logits = torch.cat([public_logits, logits.detach().cpu()], dim=0)

        # Information entropy
        entropy = (-public_logits * np.log(public_logits + 10 ** (-8))).sum(axis=-1)

        # # Sampling knowledge transfer data
        _, indices = torch.topk(entropy, N_query, largest=True)

        self.model.cpu()
        return indices

    def ensemble_logits(self, local_logits, local_weights=None):
        ensemble_logits = 0
        for k in range(len(local_logits)):
            if local_weights is not None:
                ensemble_logits += local_weights[k] * local_logits[k]
            else:
                ensemble_logits += (1 / len(local_logits)) * local_logits[k]
        return ensemble_logits

    def global_distillation(self, ensemble_logits, indices=None):
        self.model.to(self.device)

        if indices is None:
            public_loader = DataLoader(self.public_dataset, shuffle=False, batch_size=self.batch_size)
            logits_loader = DataLoader(ensemble_logits, shuffle=False, batch_size=self.batch_size)
        else:
            public_dataset = Subset(self.public_dataset, indices=indices)
            public_loader = DataLoader(public_dataset, shuffle=False, batch_size=self.batch_size)
            logits_loader = DataLoader(ensemble_logits, shuffle=False, batch_size=self.batch_size)
        test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size)
        optimizer = init_optimizer(self.optimizer_type, self.model, self.lr, self.weight_decay, self.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer, self.epochs)

        best_acc = -1
        for epoch in range(1, self.epochs + 1):
            # Training
            self.model.train()
            train_loss = 0
            train_acc = 0
            for (images, _), ensemble_logits in zip(public_loader, logits_loader):
                images, ensemble_logits = images.to(self.device), ensemble_logits.to(self.device)
                logits = self.model(images)
                loss = F.kl_div(F.log_softmax(logits / self.temperature, dim=-1), ensemble_logits, reduction='batchmean') * (self.temperature ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                labels = torch.argmax(ensemble_logits, dim=-1)
                train_acc += accuracy_score(y_pred=preds.detach().cpu(), y_true=labels.detach().cpu())
            scheduler.step()
            train_loss /= len(public_loader)
            train_acc /= len(public_loader)

            # Testing
            self.model.eval()
            test_loss = 0
            test_acc = 0
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    logits = self.model(images)
                loss = self.criterion(logits, labels)

                test_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                test_acc += accuracy_score(y_pred=preds.detach().cpu(), y_true=labels.detach().cpu())
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), '{}/{}-{}.pkl'.format(self.args.output_dir, self.name, self.model_type))
            logging.info("Epoch: {}/{}\tTrain Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, *Best Acc: {:.4f}".format(epoch, self.epochs, train_loss, train_acc, test_loss, test_acc, best_acc))

        self.model.cpu()
        return best_acc


class Client:
    def __init__(self, args, id, model_type, train_dataset=None, test_dataset=None, public_dataset=None):
        self.args = args
        self.id = id
        self.name = 'client{}'.format(id)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.public_dataset = public_dataset

        self.device = args.device
        self.batch_size = args.batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = args.E_k
        self.dis_epochs = args.E2_k
        self.lr = args.lr_k
        self.model_type = model_type
        # self.num_classes = args.num_classes
        self.model = init_model(self.model_type, args.num_classes, self.name)
        self.optimizer_type = args.optimizer
        self.scheduler_type = args.scheduler
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.temperature = args.temperature
        self.epsilon = args.epsilon

    def fork(self, ensemble_parameters):  # for FedAvg
        self.model.load_state_dict(ensemble_parameters)

    def local_update(self, k=None, DPSGD=False):
        self.model.to(self.device)

        if k is None:
            train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size)
        else:
            indices = np.random.choice(len(self.train_dataset), k, replace=False)
            train_dataset = Subset(self.train_dataset, indices=indices)
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size)

        if DPSGD:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator

            privacy_engine = PrivacyEngine()
            self.model = ModuleValidator.fix(self.model)
            optimizer = init_optimizer(self.optimizer_type, self.model, self.lr, self.weight_decay, self.momentum)
            self.model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(module=self.model,
                                                                                           optimizer=optimizer,
                                                                                           data_loader=train_loader,
                                                                                           epochs=self.epochs,
                                                                                           target_epsilon=self.epsilon,
                                                                                           target_delta=1e-5,
                                                                                           max_grad_norm=1.0)
        else:
            optimizer = init_optimizer(self.optimizer_type, self.model, self.lr, self.weight_decay, self.momentum)

        scheduler = init_scheduler(self.scheduler_type, optimizer, self.epochs)

        best_acc = -1
        for epoch in range(1, self.epochs + 1):
            # Training
            self.model.train()
            train_loss = 0
            train_acc = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                train_acc += accuracy_score(y_pred=preds.detach().cpu(), y_true=labels.detach().cpu())
            scheduler.step()
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            # Testing
            self.model.eval()
            test_loss = 0
            test_acc = 0
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    logits = self.model(images)
                loss = self.criterion(logits, labels)

                test_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                test_acc += accuracy_score(y_pred=preds.detach().cpu(), y_true=labels.detach().cpu())
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), '{}/{}-{}.pkl'.format(self.args.output_dir, self.name, self.model_type))
            logging.info("Epoch: {}/{}\tTrain Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, *Best Acc: {:.4f}".format(epoch, self.epochs, train_loss, train_acc, test_loss, test_acc, best_acc))

        self.model.cpu()
        return best_acc

    def compute_logits(self, indices=None):
        self.model.to(self.device)

        if indices is None:
            public_loader = DataLoader(self.public_dataset, shuffle=False, batch_size=self.batch_size)

        else:
            public_dataset = Subset(self.public_dataset, indices=indices)
            public_loader = DataLoader(public_dataset, shuffle=False, batch_size=self.batch_size)

        self.model.eval()
        public_logits = None
        for images, _ in public_loader:
            images = images.to(self.device)
            with torch.no_grad():
                logits = self.model(images)
            logits = torch.softmax(logits / self.temperature, dim=-1)

            if public_logits is None:
                public_logits = logits.detach().cpu()
            else:
                public_logits = torch.cat([public_logits, logits.detach().cpu()], dim=0)

        self.model.cpu()
        return public_logits

    def perturb_logits(self, logits, mechanism='PM'):
        # apply LDP mechanism
        for i in range(len(logits)):
            max_v = torch.max(logits[i])
            min_v = torch.min(logits[i])
            logits[i] = (logits[i] - min_v) / (max_v - min_v) * 2.0 - 1.0
            if mechanism == 'PM':
                logits[i] = torch.from_numpy(PM_md(logits[i], self.epsilon))
            elif mechanism == 'Lap':
                logits[i] = Lap(logits[i], self.epsilon)
            else:
                raise ValueError(mechanism)
            logits[i] = (1.0 + logits[i]) / 2.0 * (max_v - min_v) + min_v
        return logits

    def local_distillation(self, ensemble_logits, indices=None):
        self.model.to(self.device)

        if indices is None:
            public_loader = DataLoader(self.public_dataset, shuffle=False, batch_size=self.batch_size)
            logits_loader = DataLoader(ensemble_logits, shuffle=False, batch_size=self.batch_size)
        else:
            public_dataset = Subset(self.public_dataset, indices=indices)
            public_loader = DataLoader(public_dataset, shuffle=False, batch_size=self.batch_size)
            logits_loader = DataLoader(ensemble_logits, shuffle=False, batch_size=self.batch_size)
        test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size)
        optimizer = init_optimizer(self.optimizer_type, self.model, self.lr, self.weight_decay, self.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer, self.epochs)

        best_acc = -1
        for epoch in range(1, self.dis_epochs + 1):
            # Training
            self.model.train()
            train_loss = 0
            train_acc = 0
            for (images, _), ensemble_logits in zip(public_loader, logits_loader):
                images, ensemble_logits = images.to(self.device), ensemble_logits.to(self.device)
                logits = self.model(images)
                loss = F.kl_div(F.log_softmax(logits / self.temperature, dim=-1), ensemble_logits, reduction='batchmean') * (self.temperature ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                labels = torch.argmax(ensemble_logits, dim=-1)
                train_acc += accuracy_score(y_pred=preds.detach().cpu(), y_true=labels.detach().cpu())
            scheduler.step()
            train_loss /= len(public_loader)
            train_acc /= len(public_loader)

            # Testing
            self.model.eval()
            test_loss = 0
            test_acc = 0
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    logits = self.model(images)
                loss = self.criterion(logits, labels)

                test_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                test_acc += accuracy_score(y_pred=preds.detach().cpu(), y_true=labels.detach().cpu())
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), '{}/{}-{}.pkl'.format(self.args.output_dir, self.name, self.model_type))
            logging.info("Epoch: {}/{}\tTrain Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, *Best Acc: {:.4f}".format(epoch, self.epochs, train_loss, train_acc, test_loss, test_acc, best_acc))

        self.model.cpu()
        return best_acc


# Piecewise mechanism for one-dimensional and multi-dimensional data
# z_i' = PM-ONE(z_i, ε/m)
def PM_1d(z_i, eps):
    C = (math.exp(eps / 2) + 1) / (math.exp(eps / 2) - 1)  # C=(e^(ε/2)+1)/(e^(ε/2)-1)
    l_z_i = (C + 1) * z_i / 2 - (C - 1) / 2  # L(z_i)=(C+1)*z_i/2 - (C-1)/2
    r_z_i = l_z_i + C - 1  # R(z_i)=L(z_i)+C-1
    # provide 'size' parameter in uniform() would result in a ndarray
    x = np.random.uniform(0, 1)
    threshold = math.exp(eps / 2) / (math.exp(eps / 2) + 1)  # e^(ε/2)/e^(ε/2)+1
    if x < threshold:  # v < e^(ε/2)/e^(ε/2)+1
        z_star = np.random.uniform(l_z_i, r_z_i)  # [L(z_i), R(z_i)]
    else:
        tmp_l = np.random.uniform(-C, l_z_i)  # [-C, L(z_i)]
        tmp_r = np.random.uniform(r_z_i, C)  # [R(z_i), C]
        w = np.random.randint(2)
        z_star = (1 - w) * tmp_l + w * tmp_r
    # print("PM_1d t-star: %.3f" % t_star)
    return z_star


# z' = PM(z,ε)
def PM_md(z, eps):
    n_features = len(z)
    m = max(1, min(n_features, int(eps / 2.5)))
    rand_features = np.random.randint(0, n_features, size=m)
    res = np.zeros(z.shape)
    for j in rand_features:
        res[j] = (n_features / m) * PM_1d(z[j], eps / m)  # (C/m)*PM-ONE(z_i,ε/m)
    return res


# z' = Lap(z,ε)
def Lap(z, eps):
    n_features = len(z)
    loc = 0
    scale = 2 * n_features / eps
    noise = np.random.laplace(loc, scale, z.shape)
    z_noisy = z + noise
    return z_noisy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Federated
    parser.add_argument("--algorithm", default="FedLA", type=str, choices=['FedAvg', 'FedMD', 'FedMD-LDP', 'FedMD-NFDP', 'FedMD-DPSGD', 'FedLA'], help="Type of algorithms")
    parser.add_argument("--seed", default=0, type=int, help="Random seed for initialization")
    parser.add_argument("--K", default=10, type=int, help="Number of clients: K")
    parser.add_argument("--C", default=1, type=float, help="Fraction of clients: C")
    parser.add_argument("--T", default=100, type=int, help="Number of communication rounds: T")
    # Data
    parser.add_argument("--data_dir", default="data", type=str, help="Directory of datasets")
    parser.add_argument("--dataset", default="cifar10", type=str, choices=['cifar10', 'cifar100'], help="Type of datasets")
    parser.add_argument("--batch_size", default=256, type=int, help="Input batch size")
    parser.add_argument("--partition", default="dirichlet", type=str, choices=['iid', 'dirichlet'], help="iid data or non-iid data with Dirichlet distribution")
    parser.add_argument("--alpha", default=1, type=float, help="Ratio of Dirichlet distribution")
    # Model
    parser.add_argument("--global_model", default="resnet18", type=str, choices=['resnet18', 'resnet34', 'resnet50'], help="Type of global model")
    parser.add_argument("--local_models", default="resnet18", type=str, help="Type of local model")
    # Optimization
    parser.add_argument("--optimizer", default="adam", type=str, choices=['sgd', 'adam'], help="Type of optimizer")
    parser.add_argument("--scheduler", default="cosine", type=str, choices=['step', 'multistep', 'cosine'], help="Type of scheduler")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate of the global model: η")
    parser.add_argument("--lr_k", default=0.01, type=float, help="Learning rate of the local model: η_k")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay if we apply")
    parser.add_argument("--E", default=10, type=int, help="Number of global update epochs: E")
    parser.add_argument("--E_k", default=10, type=int, help="Number of local update epochs: E_k")
    parser.add_argument("--E2_k", default=10, type=int, help="Number of local distillation epochs: E'_k")

    parser.add_argument("--temperature", default=1, type=float, help="Temperature for distillation")
    parser.add_argument("--N_query", default=20000, type=int, help="Number of query samples")
    parser.add_argument("--epsilon", default=5, type=int, help="Privacy budget")
    parser.add_argument("--k", default=3000, type=int, help="Number of private samples for NFDP")

    # Output
    parser.add_argument("--output_dir", default="runs", type=str, help="The output directory where checkpoints/results/logs will be written.")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set dir
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.algorithm)
    os.makedirs(args.output_dir, exist_ok=True)

    # Set log
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[%(levelname)s](%(asctime)s) %(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, 'log.txt')), logging.StreamHandler(sys.stdout)])

    # Set data
    train_dataset, test_dataset, public_dataset = load_dataset(args)
    local_datasets, user_cls_counts = partition_dataset(args, train_dataset)

    # Set model
    global_model = args.global_model
    local_models = args.local_models.split(',')
    if len(local_models) == 1:
        local_models = [local_models[0]] * args.K

    # Federated training
    logger.info("Algorithm: {}".format(args.algorithm))
    logger.info("Device: {}".format(args.device))
    logger.info("Dataset: {}".format(args.dataset))
    if args.partition == "iid":
        logger.info("Partition: {}".format(args.partition))
    elif args.partition == "dirichlet":
        logger.info("Partition: {}, Alpha: {}".format(args.partition, args.alpha))
    logger.info("Number of clients: {}".format(args.K))
    logger.info("Number of train datasets: {}".format([len(local_datasets[k]) for k in range(args.K)]))
    logger.info("Data statistics: %s" % str(user_cls_counts))
    logger.info("Number of test dataset: {}".format(len(test_dataset)))
    #
    logger.info("Number of public dataset: {}".format(len(public_dataset)))
    #
    logger.info("Number of communication rounds: {}".format(args.T))
    logger.info("Number of local training epochs: {}".format(args.E_k))
    if args.algorithm != 'FedAvg':
        logger.info("Number of global distillation epochs: {}".format(args.E))
        logger.info("Number of local distillation epochs: {}".format(args.E2_k))
    logger.info("Global model: {},\tLocal models: {}".format(global_model, ', '.join(local_models)))

    server = Server(args, id=0, model_type=global_model, test_dataset=test_dataset, public_dataset=public_dataset)
    clients = {k + 1: Client(args, id=k + 1, model_type=local_models[k], train_dataset=local_datasets[k], test_dataset=test_dataset, public_dataset=public_dataset) for k in range(args.K)}

    if args.algorithm == 'FedAvg':
        current_accuracies = {k: None for k in range(args.K + 1)}
        test_accuracies = pd.DataFrame(columns=range(args.K + 1))
        communication_budgets = 0
        best_acc = -1

        # initialize θ_0
        global_parameter = server.model.state_dict()

        for t in range(1, args.T + 1):
            logger.info('===============The {:d}-th round==============='.format(t))

            # the server randomly samples m = max(C*K, 1) active clients to participate federated training
            selected_clients = server.select_active_clients(args.K, args.C)

            logger.info('#################### Client Update ####################')
            local_data_sizes = []
            local_parameters = []
            for k in selected_clients:
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                """ Download: θ_k^t-1 ← θ_t-1 """
                client.fork(global_parameter)
                communication_budgets += sum(p.numel() for p in server.model.parameters())

                """ Local Update: θ_k^t ← ClientUpdate(D_k; θ_k^t-1) """
                test_acc = client.local_update()
                current_accuracies[k] = '{:.4f}'.format(test_acc)

                """ Upload """
                local_parameters.append(client.model.state_dict())  # get the parameter of clients joined in the federated training
                local_data_sizes.append(len(client.train_dataset))  # get the quantity of clients joined in the federated training for updating the clients weights
                communication_budgets += sum(p.numel() for p in client.model.parameters())
            local_weights = [local_data_size / sum(local_data_sizes) for local_data_size in local_data_sizes]

            logger.info('#################### Server Update ####################')
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

            """ Aggregation """
            global_parameter = server.merge(local_parameters, local_weights)
            server.model.load_state_dict(global_parameter)
            test_loss, test_acc = server.test()
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(server.model.state_dict(), '{}/{}-{}.pkl'.format(server.args.output_dir, server.name, server.model_type))
            current_accuracies[0] = '{:.4f}'.format(test_acc)

            logging.info("Round: {}/{}\tCommunication Budget: {:.2f}M, Test Loss: {:.4f}, Test Acc: {:.4f}, *Best Acc: {:.4f}".format(t, args.T, communication_budgets / (1000 * 1000), test_loss, test_acc, best_acc))
            test_accuracies.loc[len(test_accuracies)] = current_accuracies
            test_accuracies.to_csv(os.path.join(args.output_dir, 'test_accuracy.csv'))
            print(test_accuracies)
    elif args.algorithm == 'FedMD':
        current_accuracies = {k: None for k in range(args.K + 1)}
        test_accuracies = pd.DataFrame(columns=range(args.K + 1))
        communication_budgets = 0
        best_acc = -1

        for t in range(1, args.T + 1):
            logger.info('===============The {:d}-th round==============='.format(t))

            # the server randomly samples m = max(C*K, 1) active clients to participate federated training
            selected_clients = server.select_active_clients(args.K, args.C)

            logger.info('#################### Client Update ####################')
            local_data_sizes = []
            local_logits = []
            for k in selected_clients:
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                """ Local Update: θ_k^t ← ClientUpdate(D_k; θ_k^t-1) """
                test_acc = client.local_update()
                # current_accuracies[k] = '{:.4f}'.format(test_acc)

                """ Local Prediction: Y_k^t = F_k(X_pub; θ_k^t)"""
                logits = client.compute_logits()

                """ Upload """
                local_logits.append(logits)  # get the logits of clients joined in the federated training
                local_data_sizes.append(len(client.train_dataset))  # get the quantity of clients joined in the federated training for updating the clients weights
                communication_budgets += logits.numel()
            local_weights = [local_data_size / sum(local_data_sizes) for local_data_size in local_data_sizes]

            logger.info('#################### Server Distillation ####################')
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

            """ Aggregation: Y^t ← N_k/N * Y_k^t """
            ensemble_logits = server.ensemble_logits(local_logits, local_weights)

            """ Global Distillation: θ_^t ← ServerDistillation(X_pub, Y^t; θ^t-1) """
            test_acc = server.global_distillation(ensemble_logits)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(server.model.state_dict(), '{}/{}-{}.pkl'.format(server.args.output_dir, server.name, server.model_type))
            current_accuracies[0] = '{:.4f}'.format(test_acc)

            logger.info('#################### Client Distillation ####################')
            """ Local Distillation: θ_k^t ← ClientDistillation(X_pub, Y^t; θ_k^t) """
            for k in selected_clients:
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                test_acc = client.local_distillation(ensemble_logits)
                current_accuracies[k] = '{:.4f}'.format(test_acc)
                communication_budgets += ensemble_logits.numel()

            logging.info("Round: {}/{}\tCommunication Budget: {:.2f}M, *Best Acc: {:.4f}".format(t, args.T, communication_budgets / (1000 * 1000), best_acc))
            test_accuracies.loc[len(test_accuracies)] = current_accuracies
            test_accuracies.to_csv(os.path.join(args.output_dir, 'test_accuracy.csv'))
            print(test_accuracies)
    elif args.algorithm == 'FedMD-LDP':
        current_accuracies = {k: None for k in range(args.K + 1)}
        test_accuracies = pd.DataFrame(columns=range(args.K + 1))
        communication_budgets = 0
        best_acc = -1

        for t in range(1, args.T + 1):
            logger.info('===============The {:d}-th round==============='.format(t))

            # the server randomly samples m = max(C*K, 1) active clients to participate federated training
            selected_clients = server.select_active_clients(args.K, args.C)

            logger.info('#################### Client Update ####################')
            local_data_sizes = []
            local_logits = []
            for k in selected_clients:
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                """ Local Update: θ_k^t ← ClientUpdate(D_k; θ_k^t-1) """
                test_acc = client.local_update()
                # current_accuracies[k] = '{:.4f}'.format(test_acc)

                """ Local Prediction: Y_k^t = F_k(X_pub; θ_k^t)"""
                logits = client.compute_logits()

                #######################
                # Local Differential Privacy
                #######################
                logits = client.perturb_logits(logits, mechanism='Lap')

                """ Upload """
                local_logits.append(logits)  # get the logits of clients joined in the federated training
                local_data_sizes.append(len(client.train_dataset))  # get the quantity of clients joined in the federated training for updating the clients weights
                communication_budgets += logits.numel()
            local_weights = [local_data_size / sum(local_data_sizes) for local_data_size in local_data_sizes]

            logger.info('#################### Server Distillation ####################')
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

            """ Aggregation: Y^t ← N_k/N * Y_k^t """
            ensemble_logits = server.ensemble_logits(local_logits, local_weights)
            # print(ensemble_logits.shape)

            """ Global Distillation: θ_^t ← ServerDistillation(X_pub, Y^t; θ^t-1) """
            test_acc = server.global_distillation(ensemble_logits)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(server.model.state_dict(), '{}/{}-{}.pkl'.format(server.args.output_dir, server.name, server.model_type))
            current_accuracies[0] = '{:.4f}'.format(test_acc)

            logger.info('#################### Client Distillation ####################')
            """ Local Distillation: θ_k^t ← ClientDistillation(X_pub, Y^t; θ_k^t) """
            for k in selected_clients:
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                test_acc = client.local_distillation(ensemble_logits)
                current_accuracies[k] = '{:.4f}'.format(test_acc)
                communication_budgets += ensemble_logits.numel()

            logging.info("Round: {}/{}\tCommunication Budget: {:.2f}M, *Best Acc: {:.4f}".format(t, args.T, communication_budgets / (1000 * 1000), best_acc))
            test_accuracies.loc[len(test_accuracies)] = current_accuracies
            test_accuracies.to_csv(os.path.join(args.output_dir, 'test_accuracy.csv'))
            print(test_accuracies)
    elif args.algorithm == 'FedMD-NFDP':
        current_accuracies = {k: None for k in range(args.K + 1)}
        test_accuracies = pd.DataFrame(columns=range(args.K + 1))
        communication_budgets = 0
        best_acc = -1

        for t in range(1, args.T + 1):
            logger.info('===============The {:d}-th round==============='.format(t))

            # the server randomly samples m = max(C*K, 1) active clients to participate federated training
            selected_clients = server.select_active_clients(args.K, args.C)

            logger.info('#################### Client Update ####################')
            local_data_sizes = []
            local_logits = []
            for k in selected_clients:
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                """ Local Update: θ_k^t ← ClientUpdate(D_k; θ_k^t-1) """
                test_acc = client.local_update(k=args.k)
                # current_accuracies[k] = '{:.4f}'.format(test_acc)

                """ Local Prediction: Y_k^t = F_k(X_pub; θ_k^t)"""
                logits = client.compute_logits()

                """ Upload """
                local_logits.append(logits)  # get the logits of clients joined in the federated training
                local_data_sizes.append(len(client.train_dataset))  # get the quantity of clients joined in the federated training for updating the clients weights
                communication_budgets += logits.numel()
            local_weights = [local_data_size / sum(local_data_sizes) for local_data_size in local_data_sizes]

            logger.info('#################### Server Distillation ####################')
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

            """ Aggregation: Y^t ← N_k/N * Y_k^t """
            ensemble_logits = server.ensemble_logits(local_logits, local_weights)
            # print(ensemble_logits.shape)  # (5000,10)

            """ Global Distillation: θ_^t ← ServerDistillation(X_pub, Y^t; θ^t-1) """
            test_acc = server.global_distillation(ensemble_logits)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(server.model.state_dict(), '{}/{}-{}.pkl'.format(server.args.output_dir, server.name, server.model_type))
            current_accuracies[0] = '{:.4f}'.format(test_acc)

            logger.info('#################### Client Distillation ####################')
            """ Local Distillation: θ_k^t ← ClientDistillation(X_pub, Y^t; θ_k^t) """
            for k in selected_clients:
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                test_acc = client.local_distillation(ensemble_logits)
                current_accuracies[k] = '{:.4f}'.format(test_acc)
                communication_budgets += ensemble_logits.numel()

            logging.info("Round: {}/{}\tCommunication Budget: {:.2f}M, *Best Acc: {:.4f}".format(t, args.T, communication_budgets / (1000 * 1000), best_acc))
            test_accuracies.loc[len(test_accuracies)] = current_accuracies
            test_accuracies.to_csv(os.path.join(args.output_dir, 'test_accuracy.csv'))
            print(test_accuracies)
    elif args.algorithm == 'FedMD-DPSGD':
        current_accuracies = {k: None for k in range(args.K + 1)}
        test_accuracies = pd.DataFrame(columns=range(args.K + 1))
        communication_budgets = 0
        best_acc = -1

        for t in range(1, args.T + 1):
            logger.info('===============The {:d}-th round==============='.format(t))

            # the server randomly samples m = max(C*K, 1) active clients to participate federated training
            selected_clients = server.select_active_clients(args.K, args.C)

            logger.info('#################### Client Update ####################')
            local_data_sizes = []
            local_logits = []
            for k in selected_clients:
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                """ Local Update: θ_k^t ← ClientUpdate(D_k; θ_k^t-1) """
                test_acc = client.local_update(DPSGD=True)
                # current_accuracies[k] = '{:.4f}'.format(test_acc)

                """ Local Prediction: Y_k^t = F_k(X_pub; θ_k^t)"""
                logits = client.compute_logits()

                """ Upload """
                local_logits.append(logits)  # get the logits of clients joined in the federated training
                local_data_sizes.append(len(client.train_dataset))  # get the quantity of clients joined in the federated training for updating the clients weights
                communication_budgets += logits.numel()
            local_weights = [local_data_size / sum(local_data_sizes) for local_data_size in local_data_sizes]

            logger.info('#################### Server Distillation ####################')
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

            """ Aggregation: Y^t ← N_k/N * Y_k^t """
            ensemble_logits = server.ensemble_logits(local_logits, local_weights)
            # print(ensemble_logits.shape)

            """ Global Distillation: θ_^t ← ServerDistillation(X_pub, Y^t; θ^t-1) """
            test_acc = server.global_distillation(ensemble_logits)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(server.model.state_dict(), '{}/{}-{}.pkl'.format(server.args.output_dir, server.name, server.model_type))
            current_accuracies[0] = '{:.4f}'.format(test_acc)

            logger.info('#################### Client Distillation ####################')
            """ Local Distillation: θ_k^t ← ClientDistillation(X_pub, Y^t; θ_k^t) """
            for k in selected_clients:
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                test_acc = client.local_distillation(ensemble_logits)
                current_accuracies[k] = '{:.4f}'.format(test_acc)
                communication_budgets += ensemble_logits.numel()

            logging.info("Round: {}/{}\tCommunication Budget: {:.2f}M, *Best Acc: {:.4f}".format(t, args.T, communication_budgets / (1000 * 1000), best_acc))
            test_accuracies.loc[len(test_accuracies)] = current_accuracies
            test_accuracies.to_csv(os.path.join(args.output_dir, 'test_accuracy.csv'))
            print(test_accuracies)
    elif args.algorithm == 'FedLA':
        current_accuracies = {k: None for k in range(args.K + 1)}
        test_accuracies = pd.DataFrame(columns=range(args.K + 1))
        communication_budgets = 0
        best_acc = -1

        for t in range(1, args.T + 1):
            logger.info('===============The {:d}-th round==============='.format(t))

            # the server randomly samples m = max(C*K, 1) active clients to participate federated training
            selected_clients = server.select_active_clients(args.K, args.C)

            #######################
            # Active Data Sampling
            #######################
            indices = server.active_data_sampling(args.N_query)

            logger.info('#################### Client Update ####################')
            local_data_sizes = []
            local_logits = []
            for k in selected_clients:
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                """ Local Update: θ_k^t ← ClientUpdate(D_k; θ_k^t-1) """
                test_acc = client.local_update()
                # current_accuracies[k] = '{:.4f}'.format(test_acc)

                """ Local Prediction: Y_k^t = F_k(X_pub; θ_k^t)"""
                logits = client.compute_logits(indices=indices)
                communication_budgets += len(indices)

                #######################
                # Local Differential Privacy
                #######################
                logits = client.perturb_logits(logits)

                """ Upload """
                local_logits.append(logits)  # get the logits of clients joined in the federated training
                local_data_sizes.append(len(client.train_dataset))  # get the quantity of clients joined in the federated training for updating the clients weights
                communication_budgets += logits.numel()
            local_weights = [local_data_size / sum(local_data_sizes) for local_data_size in local_data_sizes]

            logger.info('#################### Server Distillation ####################')
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

            """ Aggregation: Y^t ← N_k/N * Y_k^t """
            ensemble_logits = server.ensemble_logits(local_logits, local_weights)
            # print(ensemble_logits.shape)

            """ Global Distillation: θ_^t ← ServerDistillation(X_pub, Y^t; θ^t-1) """
            test_acc = server.global_distillation(ensemble_logits, indices=indices)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(server.model.state_dict(), '{}/{}-{}.pkl'.format(server.args.output_dir, server.name, server.model_type))
            current_accuracies[0] = '{:.4f}'.format(test_acc)

            logger.info('#################### Client Distillation ####################')
            """ Local Distillation: θ_k^t ← ClientDistillation(X_pub, Y^t; θ_k^t) """
            for k in selected_clients:
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                test_acc = client.local_distillation(ensemble_logits, indices=indices)
                current_accuracies[k] = '{:.4f}'.format(test_acc)
                communication_budgets += ensemble_logits.numel()

            logging.info("Round: {}/{}\tCommunication Budget: {:.2f}M, *Best Acc: {:.4f}".format(t, args.T, communication_budgets / (1000 * 1000), best_acc))
            test_accuracies.loc[len(test_accuracies)] = current_accuracies
            test_accuracies.to_csv(os.path.join(args.output_dir, 'test_accuracy.csv'))
            print(test_accuracies)
    else:
        raise ValueError(args.algorithm)
