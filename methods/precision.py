import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


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
        device,
        use_multi_k,
    ):
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_ratio = k_ratio
        self.precision = precision
        self.train_noise_ind = train_noise_ind
        self.device = device
        self.use_multi_k = use_multi_k

        if self.use_multi_k:
            self.name = f"Precision(K_ratios-{self.k_ratio*100}%,Precision-{self.precision})"
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

        if self.use_multi_k:
            k_ratios = [self.k_ratio] * len(self.train_dataset.classes)
        else:
            k_ratios = [self.k_ratio]

        self.class_indices = []
        self.players = []
        for i, k_ratio in enumerate(k_ratios):
            if self.use_multi_k:
                idx = np.where(self.train_dataset.targets == i)[0]
            else:
                idx = np.arange(len(self.train_dataset))
            n_experts = len(idx)
            k = int(n_experts * k_ratio)
            player = PrecisionSelector(
                    n_experts, k, self.precision, self.train_noise_ind, self.device
                )
            self.class_indices.append(idx)
            self.players.append(player)

    def _calc_loss(self, model):
        # switch to evaluate mode
        model.eval()

        criterion = nn.CrossEntropyLoss(reduction="none")

        losses = []
        with torch.no_grad():
            for i, (images, target, indexes) in enumerate(self.fixed_train_dataloader):
                if torch.cuda.is_available():
                    images = images.to(self.device)
                    target = target.to(self.device)

                # compute output
                output = model(images)

                # compute loss
                loss = criterion(output, target)
                losses.append(loss)

        losses = torch.cat(losses, dim=0)
        return losses

    def pre_epoch_hook(self, model):
        loss = self._calc_loss(model)
        
        indices = []
        cum_losses = np.empty(len(loss))
        objectives = np.empty(len(loss))
        for i, player in enumerate(self.players):
            class_idx = self.class_indices[i]
            cum_loss, objective = player.update(loss[class_idx])
            selected_idx = torch.where(player.w)[0].cpu()
            indices.append(class_idx[selected_idx])
            cum_losses[class_idx] = cum_loss.cpu()
            objectives[class_idx] = objective.cpu()
        indices = np.concatenate(indices)
        return loss, cum_loss, objective, indices

    def loss(self, outputs, target, *args, **kwargs):
        output = outputs[0]
        loss = self.criterion(output, target)
        return [loss], [[]]


class PrecisionSelector:
    def __init__(self, n_experts, k, precision, noise_ind, device, fixed=True):
        self.n_experts = n_experts
        self.k = k
        self.precision = precision
        self.noise_ind = noise_ind
        self.device = device
        self.fixed = fixed

        self.clean_ind = [i for i in range(n_experts) if i not in noise_ind]
        self.clean_num = int(self.k * precision)
        self.noise_num = self.k - self.clean_num

        self.init()

    def select(self):
        self.w[np.random.choice(self.clean_ind, self.clean_num, replace=False)] = 1
        self.w[np.random.choice(self.noise_ind, self.noise_num, replace=False)] = 1

    def init(self):
        self.w = torch.zeros(self.n_experts, dtype=torch.bool, device=self.device)
        self.total_loss = torch.zeros(self.n_experts, device=self.device)

        self.select()

    def update(self, loss):
        self.total_loss += loss

        if not self.fixed:
            self.select()

        return self.total_loss, self.total_loss
