"""CNN для ориентации кропа. Основа — модель из sub1.ipynb (4 conv-блока 32→64→128→256 с BatchNorm).

Изменения относительно исходной версии:
1. forward возвращает логит, а не сигмоиду: вместе с BCEWithLogitsLoss это численно устойчивее.
2. Пулинг сохраняет вертикальную структуру (AdaptiveAvgPool2d((4, 1)) вместо (1, 1)):
   для ориентации важно, где по высоте находятся элементы букв (базовая линия, выносные элементы),
   а глобальный пулинг эту информацию стирает.
3. Обёртка AntisymmetricOrientation: logit(x) = g(x) - g(rot180(x)).
   Тогда p(rot180(x)) = 1 - p(x) строго, как того требует задача.
"""
import torch
import torch.nn as nn


def conv_block(c_in: int, c_out: int, pool: bool) -> nn.Sequential:
    layers = [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
              nn.BatchNorm2d(c_out),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class CNN(nn.Module):
    def __init__(self, img_channel: int = 3, dropout_p: float = 0.3, pooled_rows: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            conv_block(img_channel, 32, pool=True),
            conv_block(32, 64, pool=True),
            conv_block(64, 128, pool=True),
            conv_block(128, 256, pool=False),
            nn.AdaptiveAvgPool2d((pooled_rows, 1)), # усредняем по ширине, высоту сохраняем
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(256 * pooled_rows, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x).flatten(1)
        return self.fc(self.dropout(x)).squeeze(1) # логит


class AntisymmetricOrientation(nn.Module):
    """logit P(180 | x) = g(x) - g(rot180(x))."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.g = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        both = self.g(torch.cat([x, torch.flip(x, dims=(-2, -1))]))
        g_x, g_rot = both.chunk(2)
        return g_x - g_rot


class TimmBackbone(nn.Module):
    """Лёгкий предобученный на ImageNet бэкбон из timm (например, mobilenetv3_small_050) + линейная голова.

    Как и в CNN, по высоте оставляем 2 строки признаков, чтобы не терять «верх/низ».
    """

    def __init__(self, name: str, pretrained: bool = True, dropout_p: float = 0.2):
        super().__init__()
        import timm
        self.body = timm.create_model(name, pretrained=pretrained, num_classes=0)
        channels = self.body.feature_info[-1]["num_chs"]
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(channels * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.body.forward_features(x) # B x C x h x w, h = H / 32
        f = f.mean(dim=3, keepdim=True) # усредняем по ширине
        f = nn.functional.interpolate(f.float(), size=(2, 1), mode="area") if f.shape[2] >= 2 \
            else f.float().repeat(1, 1, 2, 1)
        return self.fc(self.dropout(f.flatten(1))).squeeze(1)


def build_model(arch: str = "cnn", pretrained: bool = False, **kwargs) -> AntisymmetricOrientation:
    backbone = CNN(**kwargs) if arch == "cnn" else TimmBackbone(arch, pretrained=pretrained, **kwargs)
    return AntisymmetricOrientation(backbone)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
