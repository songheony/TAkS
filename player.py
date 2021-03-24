import torch
import numpy as np


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
        self.total_loss = np.zeros(self.n_experts)

        self.select()

    def update(self, outputs, preds, targets):
        correct = torch.ones_like(preds)
        correct[preds != targets] = -1
        probality = torch.max(outputs, dim=1)[0]
        loss = ((1 - correct * probality) / 2).cpu().numpy()

        self.total_loss += loss

        if not self.fixed:
            self.select()

        return loss, self.total_loss, self.total_loss


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
        self.w = np.zeros(self.n_experts)
        self.w[np.random.choice(self.n_experts, self.k, replace=False)] = 1
        self.total_loss = np.zeros(self.n_experts)

    def update(self, outputs, preds, targets):
        correct = torch.ones_like(preds)
        correct[preds != targets] = -1
        probality = torch.max(outputs, dim=1)[0]
        loss = ((1 - correct * probality) / 2).cpu().numpy()
        if self.use_total:
            self.total_loss += loss

            if self.lr > 0:
                noise = np.random.randn(self.n_experts) * self.lr
                objective = self.total_loss + noise
            else:
                objective = self.total_loss
        else:
            objective = loss

        idx = np.argpartition(objective, self.k)[: self.k]

        self.w[:] = 0
        self.w[idx] = 1

        return loss, self.total_loss, objective


if __name__ == "__main__":
    n_experts = 50000
    k = 40000
    T = 1000
    lr_ratio = 1e-5
    player = Player(n_experts, k, T, lr_ratio)

    losses = np.random.randn(T, n_experts)
    losses += np.array(range(n_experts)) / n_experts

    for i in range(T):
        loss = losses[i]
        player.update(loss)
        print(f"Weight: {player.w}, Loss: {loss}")
