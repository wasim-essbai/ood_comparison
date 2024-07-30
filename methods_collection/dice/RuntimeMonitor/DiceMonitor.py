import torch
import numpy as np


class DiceMonitor(object):
    def __init__(self, sparsity_parameter, device):
        self.p = sparsity_parameter
        self.mean_act = None
        self.masked_w = None
        self.setup_flag = False
        self.device = device

    def setup(self, model, set_loader):
        if not self.setup_flag:
            activation_log = []
            model.eval()

            with torch.no_grad():
                for data, target in set_loader:
                    data = data.to(self.device)
                    data = data.float()

                    _, feature = model(data, return_feature=True)
                    activation_log.append(feature.data.cpu().numpy())

            activation_log = np.concatenate(activation_log, axis=0)
            self.mean_act = activation_log.mean(0)
            self.setup_flag = True

    def calculate_mask(self, w):
        contrib = self.mean_act[None, :] * w.data.squeeze().cpu().numpy()
        threshold = np.percentile(contrib, self.p)
        mask = torch.Tensor((contrib > threshold)).cuda()
        self.masked_w = w * mask

    @torch.no_grad()
    def process_input(self, model, data):
        fc_weight, fc_bias = model.get_fc()
        if self.masked_w is None:
            self.calculate_mask(torch.from_numpy(fc_weight).cuda())
        _, feature = model(data, return_feature=True)
        vote = feature[:, None, :] * self.masked_w
        output = vote.sum(2) + torch.from_numpy(fc_bias).cuda()
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        energy_conf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energy_conf
