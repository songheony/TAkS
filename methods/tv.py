import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Dirichlet, Categorical


default_activation = lambda t: F.softmax(t, dim=1)


class TV:
    def __init__(
        self,
        train_dataloader,
        epochs,
        device,
        dataset_name,
        transition_type,
        regularization_type,
    ):
        self.train_dataloader = train_dataloader
        self.epochs = epochs
        self.device = device
        self.dataset_name = dataset_name
        self.transition_type = transition_type
        self.regularization_type = regularization_type

        self.name = f"TV(Trans-{self.transition_type},Regul-{self.regularization_type})"
        self.num_models = 1
        self._config()

    def _config(self):
        num_iter_total = len(self.train_dataloader) * self.epochs
        num_iter_warmup = len(self.train_dataloader) * 10
        num_classes = len(self.train_dataloader.dataset.classes)

        if self.transition_type == "none":
            self.transition = NoTransition()
        elif self.transition_type == "categorical":
            diagonal = np.log(0.5)
            off_diagonal = np.log(0.5 / (num_classes - 1))
            lr = 5e-3
            self.transition = categorical_transition(
                self.device,
                num_classes,
                num_iter_warmup,
                num_iter_total,
                diagonal,
                off_diagonal,
                lr,
            )
        elif self.transition_type == "dirichlet":
            if self.dataset_name == "mnist":
                diagonal = 10.0
            elif self.dataset_name == "clothing1m":
                diagonal = 1.0
            else:
                diagonal = 100.0
            off_diagonal = 0.0
            betas = (0.999, 0.01)
            self.transition = dirichlet_transition(
                self.device, num_classes, diagonal, off_diagonal, betas
            )

        if self.regularization_type == "none":
            self.regularization = no_regularization
        elif self.regularization_type == "tv":
            self.regularization = tv_regularization(self.train_dataloader.batch_size)

        self.gamma = 0.1

    def loss(self, outputs, target, *args, **kwargs):
        losses = []
        for output in outputs:
            loss = self.transition.loss(
                output, target
            ) - self.gamma * self.regularization(output)
            self.transition.update(output, target)
            losses.append(loss)

        selected_inds = [[] for _ in range(len(outputs))]
        return losses, selected_inds


def indirect_observation_loss(transition_matrix, activation=None):
    if activation is None:
        activation = lambda t: F.softmax(t, dim=1)

    def loss(t, y):
        p_z = activation(t)
        p_y = p_z @ transition_matrix.to(y.device)
        return F.nll_loss(torch.log(p_y + 1e-32), y)

    return loss


def confusion_matrix(v1, v2, n1=None, n2=None):
    if n1 is None:
        n1 = v1.max().item() + 1
    if n2 is None:
        n2 = v2.max().item() + 1
    matrix = torch.zeros(n1, n2).long().to(v1.device)
    pairs, counts = torch.unique(torch.stack((v1, v2)), dim=1, return_counts=True)
    matrix[pairs[0], pairs[1]] = counts
    return matrix


class Transition:
    @property
    def params(self):
        return None

    @property
    def matrix(self):
        return None

    def update(self, x, y):
        pass

    def loss(self, x, y):
        return None


class NoTransition:
    def __init__(self, loss=None):
        self.loss = F.cross_entropy if loss is None else loss


class FixedTransition:
    def __init__(self, matrix):
        self.matrix = matrix
        self.loss = indirect_observation_loss(matrix)


class CategoricalTransition:
    def __init__(
        self,
        init_matrix,
        optimizer,
        scheduler=None,
        activation_output=None,
        activation_matrix=None,
    ):
        self.logits = nn.Parameter(init_matrix, requires_grad=True)
        self.optimizer = optimizer([self.logits])
        self.scheduler = scheduler(self.optimizer) if scheduler is not None else None
        self.activation_output = (
            default_activation if activation_output is None else activation_output
        )
        self.activation_matrix = (
            default_activation if activation_matrix is None else activation_matrix
        )

    @property
    def params(self):
        return self.logits

    @property
    def matrix(self):
        return self.activation_matrix(self.logits)

    def loss(self, t, y):
        self.optimizer.zero_grad()  # no accumulated gradient
        p_z = self.activation_output(t)
        p_y = p_z @ self.matrix
        return F.nll_loss(torch.log(p_y + 1e-32), y)

    def update(self, x):
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()


class DirichletTransition:
    def __init__(
        self,
        init_matrix,
        betas,
        activation_output=None,
    ):
        self.concentrations = init_matrix
        self.betas = betas
        self.activation_output = (
            default_activation if activation_output is None else activation_output
        )

    @property
    def params(self):
        return self.concentrations

    @property
    def matrix(self):
        return self.concentrations / self.concentrations.sum(dim=1, keepdim=True)

    def _sample(self):
        return torch.stack([Dirichlet(c).sample() for c in self.concentrations])

    def loss(self, t, y):
        p_z = self.activation_output(t)
        p_y = p_z @ self._sample()
        return F.nll_loss(torch.log(p_y + 1e-32), y)

    def update(self, t, y):
        num_classes = self.concentrations.shape[0]
        # z = t.detach().argmax(dim=1)  # simplified version using argmax
        z = Categorical(probs=self.activation_output(t.detach())).sample()
        m = confusion_matrix(z, y, n1=num_classes, n2=num_classes)
        self.concentrations *= self.betas[0]  # decay
        self.concentrations += self.betas[1] * m  # update


def no_regularization(x):
    return 0


def tv_regularization(num_pairs, activation_output=None):
    activation_output = (
        default_activation if activation_output is None else activation_output
    )

    def reg(t: torch.Tensor):
        p = activation_output(t)
        idx1, idx2 = torch.randint(0, t.shape[0], (2, num_pairs)).to(t.device)
        tv = 0.5 * (p[idx1] - p[idx2]).abs().sum(dim=1).mean()
        return tv

    return reg


def diag_matrix(n: int, diagonal: float, off_diagonal: float) -> torch.Tensor:
    return off_diagonal * torch.ones(n, n) + (diagonal - off_diagonal) * torch.eye(n, n)


def categorical_transition(
    device, num_classes, num_iter_warmup, num_iter_total, diagonal, off_diagonal, lr
):
    init_matrix = diag_matrix(
        num_classes, diagonal=diagonal, off_diagonal=off_diagonal
    ).to(device)
    optim_matrix = lambda params: optim.Adam(params, lr=lr)
    lr_lambda = lambda i: np.interp(
        [i], [0, num_iter_warmup, num_iter_total], [0, 1, 0]
    )[0]
    sched_matrix = lambda optimizer: optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda
    )
    return CategoricalTransition(init_matrix, optim_matrix, sched_matrix)


def dirichlet_transition(device, num_classes, diagonal, off_diagonal, betas):
    init_matrix = diag_matrix(
        num_classes, diagonal=diagonal, off_diagonal=off_diagonal
    ).to(device)
    return DirichletTransition(init_matrix, betas)
