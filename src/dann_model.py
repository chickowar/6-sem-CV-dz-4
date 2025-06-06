import torch
import torch.nn as nn
import torch.nn.functional as F


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5),  # [B, 32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),      # [B, 32, 14, 14]
            nn.Conv2d(32, 48, 5), # [B, 48, 10, 10]
            nn.ReLU(),
            nn.MaxPool2d(2)       # [B, 48, 5, 5]
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), -1)  # [B, 48*5*5]


class LabelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(48*5*5, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.classifier(x)


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(48*5*5, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x, alpha):
        x = GRL.apply(x, alpha)
        return self.discriminator(x)
