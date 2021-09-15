import os
import numpy as np

import torch
import torch.nn.functional as F


class F_correction:
    def __init__(
        self,
        classifier,
        dataset_name,
        dataloader,
        device,
    ):
        self.classifier = classifier
        self.dataset_name = dataset_name
        self.dataloader = dataloader
        self.device = device

        self.name = "F-correction"
        self.num_models = 1
        self._config()

    def _config(self):
        if self.dataset_name == "cifar100":
            filter_outlier = False
        else:
            filter_outlier = True

        est = NoiseEstimator(
            classifier=self.classifier, alpha=0.0, filter_outlier=filter_outlier
        )
        est.fit(self.dataloader, self.device)
        self.P_est = torch.tensor(est.predict().copy(), dtype=torch.float).to(
            self.device
        )
        del est

    def loss(self, outputs, target, *args, **kwargs):
        epsilon = 1e-7
        y_pred = outputs[0]

        y_pred = F.softmax(y_pred, dim=1)
        weighted_y_pred = torch.log(torch.matmul(y_pred, self.P_est) + epsilon)
        loss = F.nll_loss(weighted_y_pred, target)
        return [loss], [[]]


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

    def fit(self, train_dataloader, device, anchorrate=97):
        # predict probability on the fresh sample
        self.classifier.eval()

        eta_corr = []
        with torch.no_grad():
            for i, (images, target, indexes) in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    images = images.to(device)

                # compute output
                output = self.classifier(images)
                output = F.softmax(output, dim=1).detach()
                eta_corr.append(output)

        eta_corr = torch.cat(eta_corr, dim=0)
        eta_corr = eta_corr.cpu().numpy()

        c = len(train_dataloader.dataset.classes)
        T = np.empty((c, c))

        # find a 'perfect example' for each class
        for i in np.arange(c):

            if not self.filter_outlier:
                idx_best = np.argmax(eta_corr[:, i])
            else:
                eta_thresh = np.percentile(eta_corr[:, i], anchorrate, interpolation="higher")
                robust_eta = eta_corr[:, i]
                robust_eta[robust_eta >= eta_thresh] = 0.0
                idx_best = np.argmax(robust_eta)

            T[i, :] = eta_corr[idx_best, :]

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
