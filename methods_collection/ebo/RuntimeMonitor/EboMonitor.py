import torch


class EboMonitor(object):
    def __init__(self, temperature):
        self.temperature = temperature

    @torch.no_grad()
    def process_input(self, model, data):
        output = model(data)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
        return pred, conf
