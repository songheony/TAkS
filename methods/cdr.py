import torch


class CDR:
    def __init__(self, clip):
        self.clip = clip

        self.name = f"CDR(Clip-{self.clip})"
        self.num_models = 1
        self.nonzero_ratio = self.clip

    def post_backward_hook(self, model):
        to_concat_g = []
        to_concat_v = []
        for name, param in model.named_parameters():
            if param.dim() in [2, 4]:
                to_concat_g.append(param.grad.data.view(-1))
                to_concat_v.append(param.data.view(-1))
        all_g = torch.cat(to_concat_g)
        all_v = torch.cat(to_concat_v)
        metric = torch.abs(all_g * all_v)
        num_params = all_v.size(0)
        nz = int(self.nonzero_ratio * num_params)
        top_values, _ = torch.topk(metric, nz)
        thresh = top_values[-1]

        for name, param in model.named_parameters():
            if param.dim() in [2, 4]:
                mask = (torch.abs(param.data * param.grad.data) >= thresh).type(
                    torch.cuda.FloatTensor
                )
                mask = mask * self.clip
                param.grad.data = mask * param.grad.data

    def loss(self, outputs, target, *args, **kwargs):
        output = outputs[0]
        loss = self.criterion(output, target)
        return [loss], [[]]
