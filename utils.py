import csv
import random
import pickle
import numpy as np
from numpy.testing import assert_array_almost_equal

import torch
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def predict(train_loader, model, device, softmax=True):
    # switch to evaluate mode
    model.eval()

    outputs = []
    preds = []
    targets = []
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            if len(data) == 3:
                images, target, _ = data
            elif len(data) == 2:
                images, target = data

            if torch.cuda.is_available():
                images = images.to(device)
                target = target.to(device)

            # compute output
            output = model(images)
            if softmax:
                output = F.softmax(output, dim=1).detach()
            else:
                output = output.detach()
            outputs.append(output)

            pred = torch.argmax(output, dim=1)
            preds.append(pred)

            targets.append(target)

    outputs = torch.cat(outputs, dim=0)
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    return outputs, preds, targets


class NoiseEstimator:
    def __init__(
        self,
        classifier,
        row_normalize=True,
        alpha=0.0,
        filter_outlier=False,
        cliptozero=False,
        verbose=0,
    ):
        super().__init__()
        self.classifier = classifier
        self.row_normalize = row_normalize
        self.alpha = alpha
        self.filter_outlier = filter_outlier
        self.cliptozero = cliptozero
        self.verbose = verbose
        self.T = None

    def fit(self, train_dataloader, device):
        # predict probability on the fresh sample
        eta_corr, _, _ = predict(train_dataloader, self.classifier, device)
        eta_corr = eta_corr.cpu().numpy()

        c = len(train_dataloader.dataset.classes)
        T = np.empty((c, c))

        # find a 'perfect example' for each class
        for i in np.arange(c):

            if not self.filter_outlier:
                idx_best = np.argmax(eta_corr[:, i])
            else:
                eta_thresh = np.percentile(eta_corr[:, i], 97, interpolation="higher")
                robust_eta = eta_corr[:, i]
                robust_eta[robust_eta >= eta_thresh] = 0.0
                idx_best = np.argmax(robust_eta)

            for j in np.arange(c):
                T[i, j] = eta_corr[idx_best, j]

        self.T = T
        self.c = c

    def predict(self):
        T = self.T
        c = self.c

        if self.cliptozero:
            idx = np.array(T < 10 ** -6)
            T[idx] = 0.0

        if self.row_normalize:
            row_sums = T.sum(axis=1)
            T /= row_sums[:, np.newaxis]

        if self.verbose > 0:
            print(T)

        if self.alpha > 0.0:
            T = self.alpha * np.eye(c) + (1.0 - self.alpha) * T

        if self.verbose > 0:
            print(T)
            print(np.linalg.inv(T))

        return T


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_metric(metric, csv_path):
    with open(csv_path, "w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, metric[0].keys())
        w.writeheader()
        w.writerows(metric)


def save_best_metric(best_metric, csv_path):
    with open(csv_path, "w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, best_metric.keys())
        w.writeheader()
        w.writerow(best_metric)


def save_pickle(data, pickle_path):
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)


# flipping code from https://github.com/hongxin001/JoCoR
def multiclass_noisify(y, P):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = np.random.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1.0 - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1.0 - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1.0 - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print("Actual noise %.2f" % actual_noise)
        y_train = y_train_noisy

    return y_train, P


def noisify_multiclass_symmetric(y_train, noise, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1.0 - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1.0 - n
        P[nb_classes - 1, nb_classes - 1] = 1.0 - n

        y_train_noisy = multiclass_noisify(y_train, P=P)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print("Actual noise %.2f" % actual_noise)
        y_train = y_train_noisy

    return y_train, P


def noisify_mnist_asymmetric(y_train, noise):
    """mistakes:
        1 <- 7
        2 -> 7
        3 -> 8
        5 <-> 6
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 1 <- 7
        P[7, 7], P[7, 1] = 1.0 - n, n

        # 2 -> 7
        P[2, 2], P[2, 7] = 1.0 - n, n

        # 5 <-> 6
        P[5, 5], P[5, 6] = 1.0 - n, n
        P[6, 6], P[6, 5] = 1.0 - n, n

        # 3 -> 8
        P[3, 3], P[3, 8] = 1.0 - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print("Actual noise %.2f" % actual_noise)
        y_train = y_train_noisy

    return y_train, P


def noisify_cifar10_asymmetric(y_train, noise):
    """mistakes:
        automobile <- truck
        bird -> airplane
        cat <-> dog
        deer -> horse
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # automobile <- truck
        P[9, 9], P[9, 1] = 1.0 - n, n

        # bird -> airplane
        P[2, 2], P[2, 0] = 1.0 - n, n

        # cat <-> dog
        P[3, 3], P[3, 5] = 1.0 - n, n
        P[5, 5], P[5, 3] = 1.0 - n, n

        # automobile -> truck
        P[4, 4], P[4, 7] = 1.0 - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print("Actual noise %.2f" % actual_noise)
        y_train = y_train_noisy

    return y_train, P


def build_for_cifar100(size, noise):
    """ The noise matrix flips to the "next" class with probability 'noise'.
    """

    assert (noise >= 0.0) and (noise <= 1.0)

    P = (1.0 - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i + 1] = noise

    # adjust last row
    P[size - 1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def noisify_cifar100_asymmetric(y_train, noise):
    """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
    """
    nb_classes = 100
    P = np.eye(nb_classes)
    n = noise
    nb_superclasses = 20
    nb_subclasses = 5

    if n > 0.0:
        for i in np.arange(nb_superclasses):
            init, end = i * nb_subclasses, (i + 1) * nb_subclasses
            P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

        y_train_noisy = multiclass_noisify(y_train, P=P)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print("Actual noise %.2f" % actual_noise)
        y_train = y_train_noisy

    return y_train, P


def noisify(dataset_name, nb_classes, train_labels, noise_type, noise_rate):
    if noise_type == "pairflip":
        train_noisy_labels, P = noisify_pairflip(
            train_labels, noise_rate, nb_classes=nb_classes
        )
    if noise_type == "symmetric":
        train_noisy_labels, P = noisify_multiclass_symmetric(
            train_labels, noise_rate, nb_classes=nb_classes
        )
    if noise_type == "asymmetric":
        if dataset_name == "mnist":
            train_noisy_labels, P = noisify_mnist_asymmetric(train_labels, noise_rate)
        elif dataset_name == "cifar10":
            train_noisy_labels, P = noisify_cifar10_asymmetric(train_labels, noise_rate)
        elif dataset_name == "cifar100":
            train_noisy_labels, P = noisify_cifar100_asymmetric(
                train_labels, noise_rate
            )
    return train_noisy_labels, P
