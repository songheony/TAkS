import torch


class Decouple:
    def __init__(self, criterion):
        self.criterion = criterion

        self.name = "Decouple"
        self.num_models = 2

    def loss(self, outputs, target, *args, **kwargs):
        output1, output2 = outputs
        pred1 = torch.argmax(output1.data, dim=1)
        pred2 = torch.argmax(output2.data, dim=1)
        ind_update = (pred1 != pred2).nonzero(as_tuple=True)[0]

        if len(ind_update) > 0:
            loss1 = self.criterion(output1[ind_update], target[ind_update])
            loss2 = self.criterion(output2[ind_update], target[ind_update])
        else:
            loss1, loss2 = None, None

        return [loss1, loss2], [ind_update, ind_update]
