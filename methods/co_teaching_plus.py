import numpy as np
import torch
import torch.nn.functional as F

from methods.co_teaching import Co_teaching


class Co_teaching_plus:
    def __init__(self, dataset_name, forget_rate, epochs):
        self.dataset_name = dataset_name
        self.forget_rate = forget_rate
        self.epochs = epochs

        self.name = f"Co-teaching+(Forget-{self.forget_rate * 100}%)"
        self.num_models = 2
        self._config()

    def _config(self):
        if self.dataset_name == "mnist":
            self.init_epoch = 0
        elif self.dataset_name == "cifar10":
            self.init_epoch = 20
        elif self.dataset_name == "cifar100":
            self.init_epoch = 5
        elif self.init_epoch == "tiny-imagenet":
            self.init_epoch = 100
        else:
            self.init_epoch = 5

        self.co_teaching = Co_teaching(self.dataset_name, self.forget_rate, self.epochs)

        self.step = 0

    def loss(self, outputs, target, epoch, ind, *args, **kwargs):
        self.step += 1
        if epoch < self.init_epoch:
            losses, ind_updates = self.co_teaching.loss(outputs, target, epoch)
        else:
            losses, ind_updates = self.loss_coteaching_plus(
                outputs,
                target,
                ind,
            )
        return losses, ind_updates

    def loss_coteaching_plus(self, outputs, target, ind):
        logits, logits2 = outputs

        outputs = F.softmax(logits, dim=1)
        outputs2 = F.softmax(logits2, dim=1)

        _, pred1 = torch.max(logits.data, 1)
        _, pred2 = torch.max(logits2.data, 1)

        pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

        logical_disagree_id = np.zeros(target.size(), dtype=bool)
        disagree_id = []
        for idx, p1 in enumerate(pred1):
            if p1 != pred2[idx]:
                disagree_id.append(idx)
                logical_disagree_id[idx] = True

        temp_disagree = ind * logical_disagree_id.astype(np.int64)
        ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
        try:
            assert ind_disagree.shape[0] == len(disagree_id)
        except Exception:
            disagree_id = disagree_id[: ind_disagree.shape[0]]

        _update_step = np.logical_or(logical_disagree_id, self.step < 5000).astype(
            np.float32
        )
        update_step = torch.from_numpy(_update_step).to(logits.device)

        if len(disagree_id) > 0:
            update_labels = target[disagree_id]
            update_outputs = outputs[disagree_id]
            update_outputs2 = outputs2[disagree_id]

            losses, ind_updates = self.co_teaching.loss(
                [update_outputs, update_outputs2], update_labels
            )
            loss_1, loss_2 = losses
            ind_1_update, ind_2_update = ind_updates

            ind_1_update = [disagree_id[i] for i in ind_1_update]
            ind_2_update = [disagree_id[i] for i in ind_2_update]
        else:
            update_labels = target
            update_outputs = outputs
            update_outputs2 = outputs2

            cross_entropy_1 = F.cross_entropy(update_outputs, update_labels)
            cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels)

            loss_1 = torch.sum(update_step * cross_entropy_1) / target.size()[0]
            loss_2 = torch.sum(update_step * cross_entropy_2) / target.size()[0]

            ind_1_update = list(range(len(ind)))
            ind_2_update = list(range(len(ind)))
        return [loss_1, loss_2], [ind_1_update, ind_2_update]
