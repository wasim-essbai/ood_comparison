from torch import nn


class MnistAnnModel(nn.Module):

    def __init__(self):
        super(MnistAnnModel, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(40, 20, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
        )
        self.linear1 = nn.Linear(320, 320)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(320, 160)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(160, 80)
        self.activation3 = nn.ReLU()
        self.linear4 = nn.Linear(80, 40)
        self.activation4 = nn.ReLU()
        self.linear5 = nn.Linear(40, 10)
        self.activation5 = nn.Softmax(dim=1)
        self.observed_layer = None

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature1 = self.conv_net(x)

        out = self.linear1(feature1)
        feature2 = self.activation1(out)

        out = self.linear2(feature2)
        feature3 = self.activation2(out)

        out = self.linear3(feature3)
        out = self.activation3(out)

        out = self.linear4(out)
        feature = self.activation4(out)

        self.observed_layer = out

        logits_cls = self.linear5(out)
        out = self.activation5(out)

        feature_list = [feature1, feature2, feature3, feature]

        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    def get_fc(self):
        fc = self.linear5
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()
