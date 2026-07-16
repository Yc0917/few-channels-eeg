"""用于跨被试二阶 MAML 的共享双分支 EEGNet。"""

from typing import Tuple

import torch
import torch.nn as nn

from .EEGNet_SpatialTransformer import ChannelSpatialTransformer


TRANSFORMER_BRANCH = "transformer"
CONV_BRANCH = "conv"


class MetaEEGNet(nn.Module):
    """共享时间骨干，并在末端切换空间 Transformer 或空间卷积。"""

    def __init__(self,
                 n_classes: int = 4,
                 samples: int = 1001,
                 f1: int = 8,
                 f2: int = 16,
                 kernel_length: int = 64,
                 dropout_rate: float = 0.5,
                 transformer_heads: int = 2,
                 transformer_layers: int = 2,
                 transformer_ff_ratio: int = 4):
        super(MetaEEGNet, self).__init__()

        if samples < 32:
            raise ValueError("输入时间点数必须大于等于32。")
        if f2 % transformer_heads != 0:
            raise ValueError("F2必须能被Transformer注意力头数整除。")

        self.n_classes = n_classes
        self.samples = samples
        self.f1 = f1
        self.f2 = f2

        # 两个分支共用时间卷积与第一次时间降采样。
        self.shared_temporal = nn.Sequential(
            nn.ZeroPad2d((kernel_length // 2 - 1,
                          kernel_length - kernel_length // 2,
                          0,
                          0)),
            nn.Conv2d(in_channels=1,
                      out_channels=f1,
                      kernel_size=(1, kernel_length),
                      bias=False),
            # 二阶MAML中禁用运行统计，避免functional_call修改共享缓冲区。
            nn.BatchNorm2d(f1, track_running_stats=False),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate))

        # 可分离卷积只沿时间轴运算，能够同时处理22通道与3通道输入。
        self.shared_separable = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=f1,
                      out_channels=f1,
                      kernel_size=(1, 16),
                      groups=f1,
                      bias=False),
            nn.Conv2d(in_channels=f1,
                      out_channels=f2,
                      kernel_size=(1, 1),
                      bias=False),
            nn.BatchNorm2d(f2, track_running_stats=False),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate))

        # Inner loop 私有空间模块：对22个通道执行多头自注意力。
        self.transformer_spatial = ChannelSpatialTransformer(
            input_dim=f2,
            output_dim=f2,
            channels=22,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            ff_ratio=transformer_ff_ratio,
            dropout=dropout_rate)

        # Outer loop 私有空间模块：使用3通道深度空间卷积。
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(in_channels=f2,
                      out_channels=f2,
                      kernel_size=(3, 1),
                      groups=f2,
                      bias=False),
            nn.BatchNorm2d(f2, track_running_stats=False),
            nn.ELU())

        feature_size = f2 * (samples // 32)
        self.shared_classifier = nn.Linear(feature_size, n_classes)

    def forward_features(self,
                         x: torch.Tensor,
                         branch: str) -> torch.Tensor:
        """返回指定空间分支产生的展平特征。"""
        if x.ndim != 4:
            raise ValueError(f"模型输入必须为四维张量，实际为{x.shape}。")

        expected_channels = 22 if branch == TRANSFORMER_BRANCH else 3
        if branch not in (TRANSFORMER_BRANCH, CONV_BRANCH):
            raise ValueError(f"未知分支：{branch}")
        if x.shape[2] != expected_channels:
            raise ValueError(
                f"{branch}分支期望{expected_channels}通道，实际为{x.shape[2]}。")

        output = self.shared_temporal(x)
        output = self.shared_separable(output)
        if branch == TRANSFORMER_BRANCH:
            output = self.transformer_spatial(output)
        else:
            output = self.conv_spatial(output)

        return output.reshape(output.size(0), -1)

    def forward(self, x: torch.Tensor, branch: str) -> torch.Tensor:
        features = self.forward_features(x, branch)
        expected_features = self.shared_classifier.in_features
        if features.shape[1] != expected_features:
            raise ValueError(
                f"分类器期望{expected_features}维特征，实际为{features.shape[1]}。")
        return self.shared_classifier(features)

    def inner_parameter_names(self) -> Tuple[str, ...]:
        """Inner loop 更新共享参数和 Transformer，不更新卷积分支。"""
        return tuple(
            name for name, _ in self.named_parameters()
            if not name.startswith("conv_spatial."))

    def conv_finetune_parameter_names(self) -> Tuple[str, ...]:
        """测试微调时更新共享参数和三通道卷积，不更新 Transformer。"""
        return tuple(
            name for name, _ in self.named_parameters()
            if not name.startswith("transformer_spatial."))
