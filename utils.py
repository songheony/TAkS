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
