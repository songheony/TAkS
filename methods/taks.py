import torch
import torch.nn as nn
import numpy as np


class TAkS:
    def __init__(
        self, classifier, dataloader, criterion, epochs, k_ratio, lr_ratio, device, use_auto_k=False, use_multi_k=False, use_total=True, use_noise=True,
    ):
        self.classifier = classifier
        self.dataloader = dataloader
        self.criterion = criterion
        self.epochs = epochs
        self.k_ratio = k_ratio
        self.lr_ratio = lr_ratio
        self.device = device
        self.use_auto_k = use_auto_k
        self.use_multi_k = use_multi_k
        self.use_total = use_total
        self.use_noise = use_noise

        if self.use_auto_k:
            if self.k_ratio == 0:
                self.name = f"TAkS(Auto K,LR_ratio-{self.lr_ratio},Total-{self.use_total},Noise-{self.use_noise})"
            else:
                self.name = f"TAkS(Auto K-{self.k_ratio*100}%,LR_ratio-{self.lr_ratio},Total-{self.use_total},Noise-{self.use_noise})"
        elif self.use_multi_k:
            self.name = f"TAkS(K_ratios-{self.k_ratio*100}%,LR_ratio-{self.lr_ratio},Total-{self.use_total},Noise-{self.use_noise})"
        else:
            self.name = f"TAkS(K_ratio-{self.k_ratio*100}%,LR_ratio-{self.lr_ratio},Total-{self.use_total},Noise-{self.use_noise})"
        self.num_models = 1
        self._config()

    def _predict(self, model, compute_loss=False):
        # switch to evaluate mode
        model.eval()

        c = len(self.dataloader.dataset.classes)
        losses = []
        corrects = np.zeros((c))
        total = np.zeros((c))
        with torch.no_grad():
            for images, target, indexes in self.dataloader:
                if torch.cuda.is_available():
                    images = images.to(self.device)
                    target = target.to(self.device)

                # compute output
                output = model(images)

                # compute loss
                if compute_loss:
                    loss = self.criterion_for_select(output, target)
                    losses.append(loss)
                # compute correct ratio
                else:
                    pred = output.argmax(dim=1).T
                    correct = (pred == target)

                    for i in range(c):
                        mask_class = (target == i)
                        corrects[i] += (correct * mask_class).sum()
                        total[i] += mask_class.sum()

        if compute_loss:
            losses = torch.cat(losses, dim=0)
            return losses
        else:
            return corrects / total

    def _config(self):
        self.criterion_for_select = nn.CrossEntropyLoss(reduction="none")

        if self.use_auto_k:
            k_ratios = self._predict(self.classifier, compute_loss=False) + self.k_ratio
            k_ratios = np.minimum(np.maximum(k_ratios, 0), 1)
            self.use_multi_k = True
        elif self.use_multi_k:
            k_ratios = [self.k_ratio] * len(self.dataloader.dataset.classes)
        else:
            k_ratios = [self.k_ratio]

        self.class_indices = []
        self.players = []
        for i, k_ratio in enumerate(k_ratios):
            if self.use_multi_k:
                idx = np.where(self.dataloader.dataset.targets == i)[0]
            else:
                idx = np.arange(len(self.dataloader))
            n_experts = len(idx)
            k = int(n_experts * k_ratio)
            player = Player(
                n_experts, k, self.lr_ratio, self.epochs, self.device, use_total=self.use_total, use_noise=self.use_noise
            )
            self.class_indices.append(idx)
            self.players.append(player)

    def pre_epoch_hook(self, model):
        loss = self._predict(model, compute_loss=True)
        
        selected_indicies = []
        unselected_indicies = []
        cum_losses = np.empty(len(loss))
        objectives = np.empty(len(loss))
        for i, player in enumerate(self.players):
            class_idx = self.class_indices[i]
            cum_loss, objective = player.update(loss[class_idx])
            selected_idx = torch.where(player.w)[0].cpu()
            selected_indicies.append(class_idx[selected_idx])
            unselected_idx = torch.where(~player.w)[0].cpu()
            unselected_indicies.append(class_idx[unselected_idx])
            cum_losses[class_idx] = cum_loss.cpu()
            objectives[class_idx] = objective.cpu()
        selected_indicies = np.concatenate(selected_indicies)
        unselected_indicies = np.concatenate(unselected_indicies)
        return loss.cpu(), cum_losses, objectives, selected_indicies, unselected_indicies

    def loss(self, outputs, target, *args, **kwargs):
        output = outputs[0]
        loss = self.criterion(output, target)
        return [loss], [[]]


class Player:
    def __init__(self, n_experts, k, lr_ratio, T, device, use_total=True, use_noise=True):
        self.n_experts = n_experts
        self.k = k
        self.lr_ratio = lr_ratio
        self.T = T
        self.device = device
        self.use_total = use_total
        self.use_noise = use_noise

        self.init()

    def init(self):
        self.lr = self.lr_ratio * np.sqrt(self.T)
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
