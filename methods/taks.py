import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from dataset import selected_loader


class TAkS:
    def __init__(
        self, criterion, train_dataset, batch_size, epochs, k_ratio, device, use_total=True, use_noise=True,
    ):
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_ratio = k_ratio
        self.device = device
        self.use_total = use_total
        self.use_noise = use_noise

        if isinstance(self.k_ratio, list):
            self.name = f"TAkS(K_ratio-{self.k_ratio})"
        else:
            self.name = f"TAkS(K_ratio-{self.k_ratio*100}%)"
        self.num_models = 1
        self._config()

    def _config(self):
        self.fixed_train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size * 4,
            shuffle=False,
            num_workers=16,
        )

        self.coarse_indices = []
        self.players = []
        if isinstance(self.k_ratio, list):
            for i, k_ratio in enumerate(self.k_ratio):
                assert 0 < k_ratio <= 1, "k_ratio should be less than 1 and greater than 0"
                idx = np.where(np.array(self.train_dataset.coarses) == i)[0]
                n_experts = len(idx)
                k = int(n_experts * k_ratio)
                player = Player(
                    n_experts, k, self.epochs, self.device, use_total=self.use_total, use_noise=self.use_noise
                )
                self.coarse_indices.append(idx)
                self.players.append(player)
        else:
            assert 0 < self.k_ratio <= 1, "k_ratio should be less than 1 and greater than 0"
            idx = list(range(len(self.train_dataset)))
            n_experts = len(idx)
            k = int(n_experts * self.k_ratio)
            player = Player(
                n_experts, k, self.epochs, self.device, use_total=self.use_total, use_noise=self.use_noise
            )
            self.coarse_indices.append(idx)
            self.players.append(player)

        self.coarse_indices = np.array(self.coarse_indices)

    def pre_epoch_hook(self, train_dataloader, model):
        loss = calc_loss(self.fixed_train_dataloader, model, self.device)
        
        indices = []
        cum_losses = torch.empty(len(loss), device=self.device)
        objectives = torch.empty(len(loss), device=self.device)
        for i, player in enumerate(self.players):
            coarse_idx = self.coarse_indices[i]
            cum_loss, objective = player.update(loss[coarse_idx])
            selected_idx = torch.where(player.w)[0]
            indices.append(selected_idx)
            cum_losses[coarse_idx] = cum_loss
            objectives[coarse_idx] = objective
        indices = torch.cat(indices).cpu().numpy()
        cum_losses = cum_losses.cpu().numpy()
        objectives = objectives.cpu().numpy()
        selected_dataloader = selected_loader(train_dataloader, indices)
        return loss, cum_loss, objective, [indices], selected_dataloader

    def loss(self, outputs, target, *args, **kwargs):
        output = outputs[0]
        loss = self.criterion(output, target)
        return [loss], [[]]


class Player:
    def __init__(self, n_experts, k, T, device, use_total=True, use_noise=True):
        self.n_experts = n_experts
        self.k = k
        self.T = T
        self.device = device
        self.use_total = use_total
        self.use_noise = use_noise

        self.init()

    def init(self):
        self.lr = np.sqrt(self.T)
        self.w = torch.zeros(self.n_experts, dtype=torch.bool, device=self.device)
        self.w[np.random.choice(self.n_experts, self.k, replace=False)] = True
        self.total_loss = torch.zeros(self.n_experts, device=self.device)

    def update(self, loss):
        if self.use_total:
            self.total_loss += loss

            if self.use_noise:
                noise = torch.randn(self.n_experts, device=self.device) * self.lr
                objective = self.total_loss + noise
            else:
                objective = self.total_loss
        else:
            objective = loss

        _, idx = torch.topk(objective, self.k, largest=False, sorted=False)

        self.w[:] = False
        self.w[idx] = True

        return self.total_loss, objective


def calc_loss(train_loader, model, device):
    # switch to evaluate mode
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="none")

    losses = []
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            if len(data) == 3:
                images, target, _ = data
            elif len(data) == 2:
                images, target = data

            if torch.cuda.is_available():
                images = images.to(device)
                target = target.to(device)

            # compute output
            output = model(images)

            # compute loss
            loss = criterion(output, target)
            losses.append(loss)

    losses = torch.cat(losses, dim=0)
    return losses
