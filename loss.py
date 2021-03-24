import numpy as np
import torch
import torch.nn.functional as F


def loss_coteaching(outputs, t, forget_rate):
    y_1, y_2 = outputs

    loss_1 = F.cross_entropy(y_1, t, reduction="none")
    ind_1_sorted = np.argsort(loss_1.cpu().data).to(y_1.device)
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction="none")
    ind_2_sorted = np.argsort(loss_2.cpu().data).to(y_2.device)

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember].cpu()
    ind_2_update = ind_2_sorted[:num_remember].cpu()
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]

    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return (
        [
            torch.sum(loss_1_update) / num_remember,
            torch.sum(loss_2_update) / num_remember,
        ],
        [ind_1_update, ind_2_update],
    )


def loss_coteaching_plus(outputs, labels, forget_rate, ind, step):
    logits, logits2 = outputs

    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id = np.zeros(labels.size(), dtype=bool)
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

    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = torch.from_numpy(_update_step).to(logits.device)

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outputs = outputs[disagree_id]
        update_outputs2 = outputs2[disagree_id]

        losses, ind_updates = loss_coteaching(
            [update_outputs, update_outputs2], update_labels, forget_rate
        )
        loss_1, loss_2 = losses
        ind_1_update, ind_2_update = ind_updates

        ind_1_update = [disagree_id[i] for i in ind_1_update]
        ind_2_update = [disagree_id[i] for i in ind_2_update]
    else:
        update_labels = labels
        update_outputs = outputs
        update_outputs2 = outputs2

        cross_entropy_1 = F.cross_entropy(update_outputs, update_labels)
        cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels)

        loss_1 = torch.sum(update_step * cross_entropy_1) / labels.size()[0]
        loss_2 = torch.sum(update_step * cross_entropy_2) / labels.size()[0]

        ind_1_update = list(range(len(ind)))
        ind_2_update = list(range(len(ind)))
    return [loss_1, loss_2], [ind_1_update, ind_2_update]


def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(
        F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduce=False
    )

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)


def loss_jocor(outputs, t, forget_rate, co_lambda=0.1):
    y_1, y_2 = outputs
    loss_pick_1 = F.cross_entropy(y_1, t, reduce=False) * (1 - co_lambda)
    loss_pick_2 = F.cross_entropy(y_2, t, reduce=False) * (1 - co_lambda)
    loss_pick = (
        loss_pick_1
        + loss_pick_2
        + co_lambda * kl_loss_compute(y_1, y_2, reduce=False)
        + co_lambda * kl_loss_compute(y_2, y_1, reduce=False)
    ).cpu()

    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    # exchange
    loss = torch.mean(loss_pick[ind_update])

    return [loss, loss], [ind_update, ind_update]


def loss_forward(outputs, target, P):
    epsilon = 1e-7
    y_pred = outputs[0]

    y_pred = F.softmax(y_pred, dim=1)
    weighted_y_pred = torch.log(torch.matmul(y_pred, P) + epsilon)
    loss = F.nll_loss(weighted_y_pred, target)
    return [loss], [[]]


def loss_decouple(outputs, target, criterion):
    output1, output2 = outputs
    pred1 = np.argmax(output1.detach().cpu().numpy(), axis=1)
    pred2 = np.argmax(output2.detach().cpu().numpy(), axis=1)
    ind_update = np.where(pred1 != pred2)[0]

    if len(ind_update) > 0:
        loss1 = criterion(output1[ind_update], target[ind_update])
        loss2 = criterion(output2[ind_update], target[ind_update])
    else:
        loss1, loss2 = None, None

    return [loss1, loss2], [ind_update, ind_update]


def loss_general(outputs, target, criterion):
    losses = []
    for output in outputs:
        loss = criterion(output, target)
        losses.append(loss)

    selected_inds = [[] for _ in range(len(outputs))]
    return losses, selected_inds
