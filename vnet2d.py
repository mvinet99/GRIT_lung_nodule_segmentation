"""2D V-Net (Milletari et al., 2016) adapted to 2D.

Residual stages with 5x5 convolutions + PReLU, strided 2x2 convs for down-sampling,
transposed 2x2 convs for up-sampling, concatenation skips. Channels 16-32-64-128-256.
Input size must be divisible by 16 (e.g. 64x64). Output: 1-channel logits.
"""
import torch
import torch.nn as nn


def _norm(c, kind):
    if kind == "batch":
        return nn.BatchNorm2d(c)
    if kind == "instance":
        return nn.InstanceNorm2d(c, affine=True)
    if kind == "group":
        return nn.GroupNorm(min(8, c), c)
    raise ValueError(kind)


class ConvStack(nn.Module):
    def __init__(self, c, n, norm):
        super().__init__()
        layers = []
        for i in range(n):
            layers += [nn.Conv2d(c, c, 5, padding=2, bias=False), _norm(c, norm)]
            if i < n - 1:
                layers.append(nn.PReLU(c))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class InputTransition(nn.Module):
    """out = PReLU(conv5x5(x) + residual). The V-Net residual repeats the input to 16 channels;
    when 16 is not a multiple of the input channels (e.g. 3 GRIT channels) a 1x1 conv projects
    the input to 16 channels instead. 1-channel models are unchanged (old checkpoints load)."""

    def __init__(self, in_ch, out_ch, norm):
        super().__init__()
        self.out_ch = out_ch
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 5, padding=2, bias=False), _norm(out_ch, norm))
        self.act = nn.PReLU(out_ch)
        self.proj = None if out_ch % in_ch == 0 else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        res = self.proj(x) if self.proj is not None else x.repeat(1, self.out_ch // x.shape[1], 1, 1)
        return self.act(self.conv(x) + res)


class DownTransition(nn.Module):
    def __init__(self, in_ch, n_convs, norm, dropout=0.0):
        super().__init__()
        out_ch = in_ch * 2
        self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 2, stride=2, bias=False),
                                  _norm(out_ch, norm), nn.PReLU(out_ch))
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.ops = ConvStack(out_ch, n_convs, norm)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        d = self.down(x)
        return self.act(self.ops(self.drop(d)) + d)


class UpTransition(nn.Module):
    def __init__(self, in_ch, out_ch, n_convs, norm, dropout=0.0):
        super().__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch // 2, 2, stride=2, bias=False),
                                _norm(out_ch // 2, norm), nn.PReLU(out_ch // 2))
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.ops = ConvStack(out_ch, n_convs, norm)
        self.act = nn.PReLU(out_ch)

    def forward(self, x, skip):
        cat = torch.cat([self.up(self.drop(x)), skip], dim=1)
        return self.act(self.ops(cat) + cat)


class OutputTransition(nn.Module):
    def __init__(self, in_ch, n_classes, norm):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, 5, padding=2, bias=False),
                                  _norm(in_ch, norm), nn.PReLU(in_ch),
                                  nn.Conv2d(in_ch, n_classes, 1))

    def forward(self, x):
        return self.conv(x)


class VNet2D(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, norm="batch", dropout=0.2):
        super().__init__()
        self.in_tr = InputTransition(in_channels, 16, norm)
        self.down32 = DownTransition(16, 1, norm)
        self.down64 = DownTransition(32, 2, norm)
        self.down128 = DownTransition(64, 3, norm, dropout)
        self.down256 = DownTransition(128, 3, norm, dropout)
        self.up256 = UpTransition(256, 256, 3, norm, dropout)
        self.up128 = UpTransition(256, 128, 3, norm, dropout)
        self.up64 = UpTransition(128, 64, 2, norm)
        self.up32 = UpTransition(64, 32, 1, norm)
        self.out_tr = OutputTransition(32, n_classes, norm)

    def forward(self, x):
        o16 = self.in_tr(x)
        o32 = self.down32(o16)
        o64 = self.down64(o32)
        o128 = self.down128(o64)
        o256 = self.down256(o128)
        x = self.up256(o256, o128)
        x = self.up128(x, o64)
        x = self.up64(x, o32)
        x = self.up32(x, o16)
        return self.out_tr(x)


if __name__ == "__main__":
    m = VNet2D()
    y = m(torch.randn(2, 1, 64, 64))
    print(y.shape, sum(p.numel() for p in m.parameters()) / 1e6, "M params")
