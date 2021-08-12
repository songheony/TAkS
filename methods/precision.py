import numpy as np
import torch

from dataset import selected_loader
from taks import TAkS, calc_loss


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

        self.name = f"Precision(K_ratio-{self.k_ratio*100}%,Precision-{self.precision})"
        self.num_models = 1
        self._config()

    def _config(self):
        self.taks = TAkS(
            self.criterion,
            self.train_dataset,
            self.batch_size,
            self.epochs,
            self.k_ratio,
            0,
            use_total=True,
        )

        n_experts = len(self.train_dataset)
        k = int(n_experts * self.k_ratio)
        self.player = PrecisionSelector(
            n_experts, k, self.precision, self.train_noise_ind
        )

    def pre_iter_hook(self, train_dataloader):
        indices = np.where(self.player.w)[0]
        selected_dataloader = selected_loader(train_dataloader, indices)
        return selected_dataloader

    def post_iter_hook(self, model, device, indices):
        loss = calc_loss(self.fixed_train_dataloader, model, device)
        cum_loss, objective = self.player.update(loss)
        inds_updates = [indices]
        return cum_loss, objective, inds_updates

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
