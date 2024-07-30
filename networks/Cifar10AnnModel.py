from torch import nn


class Cifar10AnnModel(nn.Module):

    def __init__(self):
        super(Cifar10AnnModel, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 40, kernel_size=(5,5), stride=(1,1)),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),

            nn.Conv2d(40, 20, kernel_size=(5,5), stride=(1,1)),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),

            nn.Flatten(),
        )
        self.linear1 = nn.Linear(500, 240)
        self.batchnorm1 = nn.LayerNorm(240)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(240, 84)
        self.batchnorm2 = nn.LayerNorm(84)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(84, 10)
        self.activation3 = nn.Softmax(dim=1)
        self.observed_layer = None

    def forward(self, x, return_logits=False, return_feature=False, return_feature_list=False):
        feature1 = self.conv_net(x)

        out = self.linear1(feature1)
        # out = self.batchnorm1(out)
        feature2 = self.activation1(out)
        self.dropout1(out)

        out = self.linear2(feature2)
        # out = self.batchnorm2(out)
        feature = self.activation2(out)
        self.observed_layer = feature
        self.dropout2(feature)

        logits_cls = self.linear3(feature)
        norm_logits_cls = self.activation3(logits_cls)

        if return_logits:
          out = logits_cls
        else:
          out = norm_logits_cls

        if return_feature:
            return out, feature
        elif return_feature_list:
            feature_list = [feature1, feature2, feature]
            return out, feature_list
        else:
            return out
    
    def get_fc(self):
        fc = self.linear3
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()