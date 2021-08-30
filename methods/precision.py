import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import selected_loader
from taks import calc_loss


class Precision:
    def __init__(
        self,
        criterion,
        train_dataset,
        batch_size,
        epochs,
        k_ratio,
        precision,
        train_noise_ind,
    ):
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_ratio = k_ratio
        self.precision = precision
        self.train_noise_ind = train_noise_ind

        if isinstance(self.k_ratio, list):
            self.name = f"Precision(K_ratio-{self.k_ratio},Precision-{self.precision})"
        else:
            self.name = f"Precision(K_ratio-{self.k_ratio*100}%,Precision-{self.precision})"
        self.num_models = 1
        self._config()

    def _config(self):
        self.fixed_train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size * 4,
            shuffle=False,
            num_workers=16,
        )

        if isinstance(self.k_ratio, list):
            for i, k_ratio in enumerate(self.k_ratio):
                assert 0 < k_ratio <= 1, "k_ratio should be less than 1 and greater than 0"
                idx = np.where(np.array(self.train_dataset.coarses) == i)[0]
                n_experts = len(idx)
                k = int(n_experts * k_ratio)
                player = PrecisionSelector(
                    n_experts, k, self.precision, self.train_noise_ind
                )
                self.coarse_indices.append(idx)
                self.players.append(player)
        else:
            assert 0 < self.k_ratio <= 1, "k_ratio should be less than 1 and greater than 0"
            idx = list(range(len(self.train_dataset)))
            n_experts = len(idx)
            k = int(n_experts * self.k_ratio)
            player = PrecisionSelector(
                n_experts, k, self.precision, self.train_noise_ind
            )
            self.coarse_indices.append(idx)
            self.players.append(player)

    def pre_epoch_hook(self, train_dataloader, model, device):
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
        selected_dataloader = selected_loader(train_dataloader, indices)
        return loss, cum_loss, objective, [indices], selected_dataloader

    def loss(self, outputs, target, *args, **kwargs):
        output = outputs[0]
        loss = self.criterion(output, target)
        return [loss], [[]]


class PrecisionSelector:
    def __init__(self, n_experts, k, precision, noise_ind, fixed=True):
        self.n_experts = n_experts
        self.k = k
        self.precision = precision
        self.noise_ind = noise_ind
        self.fixed = fixed

        self.clean_ind = [i for i in range(n_experts) if i not in noise_ind]
        self.clean_num = int(self.k * precision)
        self.noise_num = self.k - self.clean_num

        self.init()

    def select(self):
        self.w[np.random.choice(self.clean_ind, self.clean_num, replace=False)] = 1
        self.w[np.random.choice(self.noise_ind, self.noise_num, replace=False)] = 1

    def init(self):
        self.w = np.zeros(self.n_experts)
        self.total_loss = torch.zeros(self.n_experts)

        self.select()

    def update(self, loss):
        self.total_loss += loss

        if not self.fixed:
            self.select()

        return self.total_loss, self.total_loss
