from torch import nn


class GtsrbAnnModel(nn.Module):

    def __init__(self):
        super(GtsrbAnnModel, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 40, kernel_size=(5, 5), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(40, 20, kernel_size=(5, 5), stride=(1, 1)),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
        )
        self.linear1 = nn.Linear(500, 240)
        self.batchnorm1 = nn.LayerNorm(240)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(240, 84)
        self.batchnorm2 = nn.LayerNorm(84)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(84, 43)
        self.activation3 = nn.Softmax(dim=1)
        #self.dropout = nn.Dropout(p=0.5)
        self.observed_layer = None

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature1 = self.conv_net(x)

        out = self.linear1(feature1)
        out = self.batchnorm1(out)
        feature2 = self.activation1(out)
        # self.dropout(out)

        out = self.linear2(feature2)
        out = self.batchnorm2(out)
        feature = self.activation2(out)
        self.observed_layer = out
        # self.dropout(out)

        logits_cls = self.linear3(feature)
        #out = self.activation3(out)


        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            feature_list = [feature1, feature2, feature]
            return logits_cls, feature_list
        else:
            return logits_cls

    def get_fc(self):
        fc = self.linear3
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()
