import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


class TAkS:
    def __init__(
        self, criterion, train_dataset, batch_size, epochs, k_ratio, lr_ratio, device, dataset_name, log_dir, dataset_log_dir, model_name, seed, use_multi_k=False, use_total=True, use_noise=True,
    ):
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_ratio = k_ratio
        self.lr_ratio = lr_ratio
        self.device = device
        self.dataset_name = dataset_name
        self.log_dir = log_dir
        self.dataset_log_dir = dataset_log_dir
        self.model_name = model_name
        self.seed = seed
        self.use_multi_k = use_multi_k
        self.use_total = use_total
        self.use_noise = use_noise

        assert 0 <= k_ratio <= 1, f"{k_ratio} must be 0 <= k_ratio <= 1"

        if self.k_ratio == 0:
            self.name = f"TAkS(Auto K,LR_ratio-{self.lr_ratio})"
        elif self.use_multi_k:
            self.name = f"TAkS(K_ratios-{self.k_ratio*100}%,LR_ratio-{self.lr_ratio})"
        else:
            self.name = f"TAkS(K_ratio-{self.k_ratio*100}%,LR_ratio-{self.lr_ratio})"
        self.num_models = 1
        self._config()

    def _predict_k(self):
        if self.dataset_name == "mnist":
            anchorrate = 90
        else:
            anchorrate = 100

        root_log_dir = os.path.join(
            self.log_dir,
            self.dataset_log_dir,
            self.model_name,
            "Standard",
            str(self.seed),
        )
        standard_path = os.path.join(
            root_log_dir,
            "model0",
            "best_model.pt",
        )
        classifier = get_model(self.model_name, self.dataset_name, self.device)
        classifier.load_state_dict(torch.load(standard_path))
        classifier.eval()
        est = NoiseEstimator(classifier=model, filter_outlier=True)
        est.fit(self.fixed_train_dataloader, self.device, anchorrate=anchorrate)
        transition_matrix = est.predict()
        k_ratios = transition_matrix.diagonal()
        return k_ratios

    def _config(self):
        self.fixed_train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size * 4,
            shuffle=False,
            num_workers=16,
        )

        if self.k_ratio == 0:
            k_ratios = self._predict_k()
            self.use_multi_k = True
        elif self.use_multi_k:
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
            player = Player(
                n_experts, k, self.lr_ratio, self.epochs, self.device, use_total=self.use_total, use_noise=self.use_noise
            )
            self.class_indices.append(idx)
            self.players.append(player)

    def pre_epoch_hook(self, train_dataloader, model):
        loss = calc_loss(self.fixed_train_dataloader, model, self.device)
        
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
        return loss, cum_loss, objective, [indices]

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


def calc_loss(train_loader, model, device):
    # switch to evaluate mode
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="none")

    losses = []
    with torch.no_grad():
        for i, (images, target, indexes) in enumerate(train_loader):
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
