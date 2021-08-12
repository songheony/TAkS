import numpy as np
import torch
import torch.nn.functional as F

from methods.co_teaching import Co_teaching


class JoCoR:
    def __init__(self, dataset_name, forget_rate, epochs, co_lambda):
        self.dataset_name = dataset_name
        self.forget_rate = forget_rate
        self.epochs = epochs
        self.co_lambda = co_lambda

        self.name = f"JoCoR(Forget-{self.forget_rate * 100}%,Lambda-{self.co_lambda})"
        self.num_models = 2
        self._config()

    def _config(self):
        self.co_teaching = Co_teaching(self.dataset_name, self.forget_rate, self.epochs)

    def loss(self, outputs, target, epoch, *args, **kwargs):
        y_1, y_2 = outputs
        loss_pick_1 = F.cross_entropy(y_1, target, reduce=False) * (1 - self.co_lambda)
        loss_pick_2 = F.cross_entropy(y_2, target, reduce=False) * (1 - self.co_lambda)
        loss_pick = (
            loss_pick_1
            + loss_pick_2
            + self.co_lambda * kl_loss_compute(y_1, y_2, reduce=False)
            + self.co_lambda * kl_loss_compute(y_2, y_1, reduce=False)
        ).cpu()

        ind_sorted = np.argsort(loss_pick.data)
        loss_sorted = loss_pick[ind_sorted]

        remember_rate = 1 - self.co_teaching.forget_rate[epoch]
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update = ind_sorted[:num_remember]

        # exchange
        loss = torch.mean(loss_pick[ind_update])

        return [loss, loss], [ind_update, ind_update]


def kl_loss_compute(pred, soft_targets, reduce=True):
    kl = F.kl_div(
        F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduce=False
    )

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)
