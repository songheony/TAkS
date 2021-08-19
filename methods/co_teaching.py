import numpy as np
import torch
import torch.nn.functional as F


class Co_teaching:
    def __init__(self, dataset_name, forget_rate, epochs):
        self.dataset_name = dataset_name
        self.forget_rate = forget_rate
        self.epochs = epochs

        self.name = f"Co-teaching(Forget-{self.forget_rate * 100}%)"
        self.num_models = 2
        self._config()

    def _config(self):
        exponent = 1
        if self.dataset_name == "clothing1m":
            num_gradual = 5
        else:
            num_gradual = 10

        self.rate_schedule = np.ones(self.epochs) * self.forget_rate
        self.rate_schedule[:num_gradual] = np.linspace(
            0, self.forget_rate ** exponent, num_gradual
        )

    def loss(self, outputs, target, epoch, *args, **kwargs):
        y_1, y_2 = outputs

        loss_1 = F.cross_entropy(y_1, target, reduction="none")
        ind_1_sorted = torch.argsort(loss_1)
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, target, reduction="none")
        ind_2_sorted = torch.argsort(loss_2)

        remember_rate = 1 - self.rate_schedule[epoch - 1]
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember].cpu()
        ind_2_update = ind_2_sorted[:num_remember].cpu()
        if len(ind_1_update) == 0:
            ind_1_update = ind_1_sorted.cpu().numpy()
            ind_2_update = ind_2_sorted.cpu().numpy()
            num_remember = ind_1_update.shape[0]

        loss_1_update = F.cross_entropy(y_1[ind_2_update], target[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], target[ind_1_update])

        return (
            [
                torch.sum(loss_1_update) / num_remember,
                torch.sum(loss_2_update) / num_remember,
            ],
            [ind_1_update, ind_2_update],
        )
