import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import selected_loader


class TAkS:
    def __init__(
        self, criterion, train_dataset, batch_size, epochs, k_ratio, lr_ratio, use_total
    ):
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_ratio = k_ratio
        self.lr_ratio = lr_ratio
        self.use_total = use_total

        if isinstance(self.k_ratio, list):
            self.name = f"TAkS(K_ratio-{self.k_ratio},Lr-{self.lr_ratio})"
        else:
            self.name = f"TAkS(K_ratio-{self.k_ratio*100}%,Lr-{self.lr_ratio})"
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
                    n_experts, k, self.epochs, self.lr_ratio, use_total=self.use_total
                )
                self.coarse_indices.append(idx)
                self.players.append(player)
        else:
            assert 0 < self.k_ratio <= 1, "k_ratio should be less than 1 and greater than 0"
            idx = list(range(len(self.train_dataset)))
            n_experts = len(idx)
            k = int(n_experts * self.k_ratio)
            player = Player(
                n_experts, k, self.epochs, self.lr_ratio, use_total=self.use_total
            )
            self.coarse_indices.append(idx)
            self.players.append(player)

    def pre_epoch_hook(self, train_dataloader):
        indices = []
        for i, player in enumerate(self.players):
            idx = np.where(player.w)[0]
            coarse_idx = self.coarse_indices[i][idx]
            indices = np.concatenate((indices, coarse_idx))
        selected_dataloader = selected_loader(train_dataloader, indices)
        return selected_dataloader

    def post_epoch_hook(self, model, device):
        loss = calc_loss(self.fixed_train_dataloader, model, device)
        
        indices = []
        cum_losses = np.empty((len(loss)))
        objectives = np.empty((len(loss)))
        for i, player in enumerate(self.players):
            idx = np.where(player.w)[0]
            coarse_idx = self.coarse_indices[i][idx]
            cum_loss, objective = player.update(loss[coarse_idx])
            indices = np.concatenate((indices, coarse_idx))
            cum_losses[coarse_idx] = cum_loss
            objectives[coarse_idx] = objective

        inds_updates = [indices]
        return loss, cum_loss, objective, inds_updates

    def loss(self, outputs, target, *args, **kwargs):
        output = outputs[0]
        loss = self.criterion(output, target)
        return [loss], [[]]


class Player:
    def __init__(self, n_experts, k, T, lr_ratio, use_total=True):
        self.n_experts = n_experts
        self.k = k
        self.T = T
        self.lr_ratio = lr_ratio
        self.use_total = use_total

        self.init()

    def init(self):
        self.lr = np.sqrt(self.k * self.T) * self.lr_ratio
        self.w = np.zeros(self.n_experts, dtype=bool)
        self.w[np.random.choice(self.n_experts, self.k, replace=False)] = True
        self.total_loss = torch.zeros(self.n_experts)

    def update(self, loss):
        if self.use_total:
            self.total_loss += loss

            if self.lr > 0:
                noise = np.random.randn(self.n_experts) * self.lr
                objective = self.total_loss + noise
            else:
                objective = self.total_loss
        else:
            objective = loss

        _, idx = torch.topk(objective, self.k, largest=False, sorted=False)

        self.w[:] = False
        self.w[idx] = True

        return self.total_loss, objective


def calc_loss(train_loader, model, criterion, device):
    # switch to evaluate mode
    model.eval()

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
