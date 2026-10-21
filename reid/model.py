import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights


class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.p = nn.Parameter(torch.tensor(3.0))
        self.necks = nn.ModuleList([nn.BatchNorm1d(2048) for _ in range(3)])
        for neck in self.necks:
            nn.init.ones_(neck.weight); nn.init.zeros_(neck.bias)
            neck.bias.requires_grad_(False)

    def pool(self, x):
        p = self.p.clamp(1, 6)
        return x.clamp_min(1e-6).pow(p).mean((2,3)).pow(1/p)

    def forward(self, x):
        fmap = self.backbone(x)
        top, bottom = fmap.chunk(2, dim=2)
        return tuple(F.normalize(neck(self.pool(part)), dim=1)
                     for neck, part in zip(self.necks, (fmap, top, bottom)))
