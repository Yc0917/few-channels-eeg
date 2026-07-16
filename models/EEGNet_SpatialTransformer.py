import torch
import torch.nn as nn


class ChannelSpatialTransformer(nn.Module):
    """在每个时间点上对 EEG 通道进行自注意力建模。"""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 channels: int,
                 num_heads: int,
                 num_layers: int,
                 ff_ratio: int,
                 dropout: float):
        super(ChannelSpatialTransformer, self).__init__()

        if output_dim % num_heads != 0:
            raise ValueError(
                f"Transformer 特征维度 {output_dim} 必须能被注意力头数 {num_heads} 整除。")
        if num_layers < 1:
            raise ValueError("Transformer 层数必须大于等于 1。")

        self.channels = channels
        self.output_dim = output_dim

        # 将每个通道的时间卷积特征映射到 Transformer 的嵌入空间。
        self.input_projection = nn.Linear(input_dim, output_dim)
        self.channel_embedding = nn.Parameter(
            torch.zeros(1, channels, output_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * ff_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers)
        self.output_norm = nn.LayerNorm(output_dim)

        nn.init.trunc_normal_(self.channel_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 形状为 [批次, input_dim, 通道数, 时间点]。

        Returns:
            形状为 [批次, output_dim, 1, 时间点] 的空间特征。
        """
        batch_size, _, channels, time_points = x.shape
        if channels != self.channels:
            raise ValueError(
                f"输入通道数为 {channels}，但模型初始化通道数为 {self.channels}。")

        # 将每个时间点的所有 EEG 通道组成一条 token 序列。
        tokens = x.permute(0, 3, 2, 1).contiguous()
        tokens = tokens.view(batch_size * time_points, channels, -1)
        tokens = self.input_projection(tokens)
        tokens = tokens + self.channel_embedding
        tokens = self.encoder(tokens)

        # 汇聚通道 token，得到与原深度空间卷积输出兼容的张量。
        spatial_feature = self.output_norm(tokens.mean(dim=1))
        spatial_feature = spatial_feature.view(
            batch_size, time_points, self.output_dim)
        spatial_feature = spatial_feature.permute(0, 2, 1).unsqueeze(2)
        return spatial_feature.contiguous()


class EEGNetSpatialTransformerFeature(nn.Module):
    """在两次时间池化之后使用通道 Transformer 的 EEGNet 特征提取器。"""

    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate: float,
                 norm_rate: float,
                 transformer_heads: int = 2,
                 transformer_layers: int = 2,
                 transformer_ff_ratio: int = 4):
        super(EEGNetSpatialTransformerFeature, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        # 移除空间深度卷积后 D 不再参与维度计算，仅保留以兼容原调用接口。
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate

        # 第一阶段提取并降采样时间特征，始终保留 EEG 通道维度。
        self.temporal_block = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2,
                          0,
                          0)),
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        # 可分离卷积只在时间轴上运算，因此不会提前压缩通道维度。
        self.separable_block = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1,
                      bias=False),
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))

        # 在分类头之前对最终低分辨率特征执行通道多头自注意力。
        self.spatial_transformer = ChannelSpatialTransformer(
            input_dim=self.F2,
            output_dim=self.F2,
            channels=self.Chans,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            ff_ratio=transformer_ff_ratio,
            dropout=self.dropoutRate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.temporal_block(x)
        output = self.separable_block(output)
        output = self.spatial_transformer(output)
        return output.reshape(output.size(0), -1)


class EEGNetSpatialTransformer(nn.Module):
    """带分类器的 EEGNet 空间 Transformer 模型。"""

    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate: float,
                 norm_rate: float,
                 transformer_heads: int = 2,
                 transformer_layers: int = 2,
                 transformer_ff_ratio: int = 4):
        super(EEGNetSpatialTransformer, self).__init__()

        self.feature_extractor = EEGNetSpatialTransformerFeature(
            n_classes=n_classes,
            Chans=Chans,
            Samples=Samples,
            kernLenght=kernLenght,
            F1=F1,
            D=D,
            F2=F2,
            dropoutRate=dropoutRate,
            norm_rate=norm_rate,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            transformer_ff_ratio=transformer_ff_ratio)

        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=F2 * (Samples // (4 * 8)),
                      out_features=n_classes,
                      bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.feature_extractor(x)
        return self.classifier_block(output)


# 保留原文件中的类名，便于替换导入路径后继续使用原训练代码。
EEGNet = EEGNetSpatialTransformer
EEGNet_feature = EEGNetSpatialTransformerFeature
