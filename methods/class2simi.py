import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from model import get_model


def norm(T):
    row_sum = np.sum(T, 1)
    T_norm = T / row_sum
    return T_norm


def class2simi(transition_matrix):
    v00 = v01 = v10 = v11 = 0
    t = transition_matrix
    num_classes = transition_matrix.shape[0]
    for i in range(num_classes):
        for j in range(num_classes):
            a = t[i][j]
            for m in range(num_classes):
                for n in range(num_classes):
                    b = t[m][n]
                    if i == m and j == n:
                        v11 += a * b
                    if i == m and j != n:
                        v10 += a * b
                    if i != m and j == n:
                        v01 += a * b
                    if i != m and j != n:
                        v00 += a * b
    simi_T = np.zeros([2, 2])
    simi_T[0][0] = v11 / (v11 + v10)
    simi_T[0][1] = v10 / (v11 + v10)
    simi_T[1][0] = v01 / (v01 + v00)
    simi_T[1][1] = v00 / (v01 + v00)
    # print(simi_T)
    return simi_T


def prepare_task_target(x, mask=None):
    mode = "hinge"
    # Convert class label to pairwise similarity
    n=x.nelement()
    assert (n-x.ndimension()+1) == n,'Dimension of Label is not right'
    expand1 = x.view(-1,1).expand(n,n)
    expand2 = x.view(1,-1).expand(n,n)
    out = expand1 - expand2    
    out[out!=0] = -1 #dissimilar pair: label=-1
    out[out==0] = 1 #Similar pair: label=1
    if mode=='cls':
        out[out==-1] = 0 #dissimilar pair: label=0
    if mode=='hinge':
        out = out.float() #hingeloss require float type
    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    train_target = out.view(-1)
    eval_target = x
    return train_target.detach(), eval_target.detach()  # Make sure no gradients


def fit(dataloader, device, model, anchorrate):
    eta_corr = []
    with torch.no_grad():
        for i, (images, target, indexes) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.to(device)

            # compute output
            output = model(images)
            output = F.softmax(output, dim=1).detach()
            eta_corr.append(output)

    eta_corr = torch.cat(eta_corr, dim=0)
    eta_corr = eta_corr.cpu().numpy()

    c = len(dataloader.dataset.classes)
    T = np.empty((c, c))

    # find a 'perfect example' for each class
    for i in np.arange(c):
        eta_thresh = np.percentile(eta_corr[:, i], anchorrate, interpolation="higher")
        robust_eta = eta_corr[:, i]
        robust_eta[robust_eta >= eta_thresh] = 0.0
        idx_best = np.argmax(robust_eta)

        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]

    return T


class Class2Simi:
    def __init__(
        self,
        loss_type,
        dataset_name,
        log_dir,
        dataset_log_dir,
        model_name,
        dataloader,
        seed,
        device,
    ):
        self.loss_type = loss_type
        self.dataset_name = dataset_name
        self.log_dir = log_dir
        self.dataset_log_dir = dataset_log_dir
        self.model_name = model_name
        self.dataloader = dataloader
        self.seed = seed
        self.device = device

        self.name = "Class2Simi"
        self.num_models = 1
        self._config()

    def _config(self):
        if self.loss_type == "forward":
            self.criterion = forward_MCL()
        else:
            self.criterion = forward_MCL()

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
        model = get_model(self.model_name, self.dataset_name, self.device)
        model.load_state_dict(torch.load(standard_path))
        model.eval()

        transition_matrix = fit(self.dataloader, self.device, model, anchorrate)
        transition_matrix = norm(transition_matrix)
        simi_T = class2simi(transition_matrix)
        self.T = torch.from_numpy(simi_T).float().to(self.device)

    def loss(self, outputs, target, *args, **kwargs):
        prob = F.softmax(outputs[0], dim=1)
        prob1, prob2 = PairEnum(prob)
        train_target, _ = prepare_task_target(target)
        loss = self.criterion(prob1, prob2, train_target, self.T)
        return [loss], [[]]


class forward_MCL(nn.Module):
    # Forward Meta Classification Likelihood (MCL)

    eps = 1e-12  # Avoid calculating log(0). Use the small value of float16.

    def __init__(self):
        super(forward_MCL, self).__init__()
        return

    def forward(self, prob1, prob2, s_label, q):
        P = prob1.mul(prob2)
        P = P.sum(1)
        P = P * q[0][0] + (1 - P) * q[1][0]
        P.mul_(s_label).add_(s_label.eq(-1).type_as(P))
        negLog_P = -P.add_(forward_MCL.eps).log_()
        return negLog_P.mean()


class reweight_MCL(nn.Module):
    # Reweight Meta Classification Likelihood (MCL)

    eps = 1e-12  # Avoid calculating log(0). Use the small value of float16.

    def __init__(self):
        super(reweight_MCL, self).__init__()
        return

    def forward(self, prob1, prob2, s_label, q):
        cleanP1 = prob1.mul(prob2)
        cleanP1 = cleanP1.sum(1)
        noiseP1 = cleanP1 * q[0][0] + (1 - cleanP1) * q[1][0]
        coef1 = cleanP1.div(noiseP1)  # coefficient for instance with \hat{Y} = 1
        coef0 = (1 - cleanP1).div(
            1 - noiseP1
        )  # coefficient for instance with \hat{Y} = 0
        coef0[s_label == 1] = coef1[
            s_label == 1
        ]  # denote the both coefficient by coef0
        coef0 = Variable(coef0, requires_grad=True)
        cleanP1.mul_(s_label).add_(s_label.eq(-1).type_as(cleanP1))
        cleanP1 = cleanP1.mul(coef0)
        negLog_P = -cleanP1.add_(reweight_MCL.eps).log_()
        return negLog_P.mean()


def PairEnum(x, mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, "Input dimension must be 2"
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1, 1).repeat(1, x.size(1))
        # dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1, x.size(1))
        x2 = x2[xmask].view(-1, x.size(1))
    return x1, x2
