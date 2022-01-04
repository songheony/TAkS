import csv
import random
import pickle
import numpy as np

import torch


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


def neg_entropy(outputs):
    probs = torch.softmax(outputs, dim=1)
    return torch.mean(torch.sum(probs.log()*probs, dim=1))


def linear_rampup(current, warm_up, lambda_u=25, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u * float(current)


def semi_loss(outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
    probs_u = torch.softmax(outputs_u, dim=1)

    Lx = -torch.mean(torch.sum(torch.nn.functional.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
    Lu = torch.mean((probs_u - targets_u)**2)

    return Lx, Lu, linear_rampup(epoch, warm_up)