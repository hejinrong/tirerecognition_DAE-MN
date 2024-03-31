# -*- coding: utf-8 -*-
# Adapted from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
from torch import nn
from mmcls.models.builder import BACKBONES

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

@BACKBONES.register_module()
class ResNet12(nn.Module):
    def __init__(
        self,
        *,
        image_size=84,
        patch_size=28,
        dim=640,
        depth=6,
        heads=16,
        mlp_dim=2048,
        pool="mean",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        # self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        return x
        # x = self.to_latent(x)
        # return self.mlp_head(x)


if __name__ == "__main__":
    v = ViT(
        image_size=84,
        patch_size=28,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )

    img = torch.randn(10, 3, 84, 84)

    preds = v(img)  # (1, 1000)

    print(preds.shape)


我的问题：给我的vit网络加入下面的注意力，请给出完整代码


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)
class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CoordAtt, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1),
            nn.BatchNorm2d(channels//reduction),
            nn.ReLU(inplace=True)
        )

        self.xfc = nn.Conv2d(channels//reduction, channels, 1)
        self.yfc = nn.Conv2d(channels//reduction, channels, 1)

    def forward(self, x):
        B, _, H, W = x.size()
        # X Avg Pool and Y Avg Pool
        xap = F.adaptive_avg_pool2d(x, (H, 1))
        yap = F.adaptive_avg_pool2d(x, (1, W))

        # Concat+Conv2d+BatchNorm+Non-linear
        mer = torch.cat([xap.transpose_(2, 3), yap], dim=3)
        fc1 = self.fc1(mer)
        
        # split
        xat, yat = torch.split(fc1, (H, W), dim=3)

        # Conv2d-Sigmoid and Conv2d-Sigmoid
        xat = torch.sigmoid(self.xfc(xat))
        yat = torch.sigmoid(self.yfc(yat))

        # Attention Multiplier
        out = x * xat * yat
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 drop_rate: float = 0.0,
                 drop_block: bool = False,
                 block_size: int = 1) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(stride)
        self.downsample = downsample

        # drop block
        self.drop_rate = drop_rate
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

        # CoordAtt 注意力模块
        self.coordatt = CoordAtt(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # 应用 CoordAtt 注意力模块
        out = self.coordatt(out)

        out += identity
        out = self.relu(out)
        out = self.max_pool(out)

        return out
    

    