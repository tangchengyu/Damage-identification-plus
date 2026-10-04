import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models

class FeatureEmbedding(nn.Module):
    """
    特征嵌入层
    将不同类型的特征转换为统一维度的嵌入向量
    """

    def __init__(self, input_dim, embedding_dim, dropout=0.1):
        """
        初始化特征嵌入层

        Args:
            input_dim: 输入特征维度
            embedding_dim: 嵌入向量维度
            dropout: Dropout率
        """
        super(FeatureEmbedding, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),  # 先扩展维度
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),  # Dropout防止过拟合
            nn.Linear(embedding_dim * 2, embedding_dim),  # 压缩到目标维度
            nn.LayerNorm(embedding_dim)  # 层归一化
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, input_dim]

        Returns:
            嵌入向量 [batch_size, embedding_dim]
        """
        return self.embedding(x)

# 移除位置编码类，因为属性特征没有前后次序之分

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder结构
    用于融合图像特征和结构化特征进行寿命预测
    """

    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        """
        初始化Transformer Decoder

        Args:
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: 层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout率
        """
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Transformer Decoder层列表，用于添加残差连接
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)

        # 多层MLP回归头
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)  # 输出单个寿命值
        )

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        前向传播

        Args:
            tgt: 目标序列（结构化特征） [batch_size, tgt_len, d_model]
            memory: 记忆序列（图像特征） [batch_size, src_len, d_model]
            tgt_mask: 目标序列掩码
            memory_mask: 记忆序列掩码

        Returns:
            预测的寿命值 [batch_size, 1]
        """
        # 不使用位置编码，因为属性特征没有前后次序之分
        output = tgt

        # 逐层应用Transformer Decoder并添加残差连接
        for i, layer in enumerate(self.decoder_layers):
            # 保存残差连接的输入
            residual = output

            # 应用Transformer层
            output = layer(
                tgt=output,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask
            )

            # 添加残差连接（每两层添加一次）
            if i > 0 and i % 2 == 1:
                output = output + residual
                output = self.layer_norm(output)

        # 取最后一个时间步的输出（或者可以使用平均池化）
        output = output.mean(dim=1)  # [batch_size, d_model] 使用平均池化而不是取最后一个

        # 多层MLP回归头
        lifetime_pred = self.output_projection(output)  # [batch_size, 1]

        return lifetime_pred

class LifetimePredictionModel(nn.Module):
    """
    齿轮寿命预测模型
    结合图像特征和结构化特征进行寿命预测
    """

    def __init__(self, config, pretrained_damage_model_path=None, pretrained_backbone=True):
        """
        初始化寿命预测模型

        Args:
            config: 配置字典
            pretrained_damage_model_path: 预训练损伤识别模型路径
        """
        super(LifetimePredictionModel, self).__init__()

        self.config = config
        model_config = config['model']

        # 图像特征提取器（冻结的损伤识别模型）
        self.image_encoder = self._load_pretrained_damage_model(pretrained_damage_model_path, pretrained_backbone)

        # 冻结图像编码器参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.image_encoder.eval()

        # 特征嵌入层
        self.size_embedding = FeatureEmbedding(
            7,  # 尺寸特征：7维
            model_config['embedding_dim'],
            model_config['dropout']
        )

        self.operation_embedding = FeatureEmbedding(
            2,  # 工况特征：2维
            model_config['embedding_dim'],
            model_config['dropout']
        )

        self.material_embedding = FeatureEmbedding(
            8,  # 材料特征：8维
            model_config['embedding_dim'],
            model_config['dropout']
        )

        # 图像特征投影层（将ResNet特征投影到统一维度）
        self.image_projection = nn.Linear(512, model_config['embedding_dim'])  # ResNet34输出512维

        # Transformer Decoder
        self.decoder = TransformerDecoder(
            d_model=model_config['embedding_dim'],
            nhead=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            dim_feedforward=model_config['hidden_dim'],
            dropout=model_config['dropout']
        )

    def _load_pretrained_damage_model(self, model_path, pretrained_backbone=True):
        """
        加载预训练的损伤识别模型

        Args:
            model_path: 模型路径

        Returns:
            损伤识别模型（仅特征提取部分）
        """
        # 创建ResNet34模型
        if model_path and not os.path.isfile(model_path):
            raise FileNotFoundError(f'Damage backbone checkpoint not found: {model_path}')
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained_backbone and not model_path else None
        resnet = models.resnet34(weights=weights)

        # 如果有预训练模型，加载权重
        if model_path:
            try:
                # 这里需要根据实际的损伤识别模型结构来调整
                # 假设损伤识别模型保存了完整的state_dict
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                # 只加载backbone部分的权重
                resnet_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('backbone.'):
                        new_key = key.replace('backbone.', '')
                        if not new_key.startswith('fc.'):
                            resnet_state_dict[new_key] = value

                expected = {k for k in resnet.state_dict() if not k.startswith('fc.')}
                if set(resnet_state_dict) != expected:
                    raise ValueError('Damage backbone tensors are incomplete or incompatible')
                resnet.load_state_dict(resnet_state_dict, strict=False)
                print(f"成功加载预训练损伤识别模型: {model_path}")
            except Exception as e:
                raise ValueError(f'Cannot load damage backbone: {e}') from e

        # 移除最后的分类层，只保留特征提取部分
        modules = list(resnet.children())[:-1]  # 移除最后的全连接层
        feature_extractor = nn.Sequential(*modules)

        return feature_extractor

    def train(self, mode=True):
        super().train(mode)
        # Frozen parameters do not freeze BatchNorm running statistics.
        self.image_encoder.eval()
        return self

    def forward(self, image, size_features, operation_features, material_features):
        """
        前向传播

        Args:
            image: 输入图像 [batch_size, 3, 224, 224]
            size_features: 尺寸特征 [batch_size, size_dim]
            operation_features: 工况特征 [batch_size, operation_dim]
            material_features: 材料特征 [batch_size, material_dim]

        Returns:
            预测的寿命值 [batch_size, 1]
        """
        batch_size = image.size(0)

        # 1. 图像特征提取
        with torch.no_grad():  # 冻结图像编码器
            image_features = self.image_encoder(image)  # [batch_size, 512, 1, 1]
            image_features = image_features.view(batch_size, -1)  # [batch_size, 512]

        # 投影到统一维度
        image_features = self.image_projection(image_features)  # [batch_size, embedding_dim]

        # 2. 结构化特征嵌入
        size_embed = self.size_embedding(size_features)  # [batch_size, embedding_dim]
        operation_embed = self.operation_embedding(operation_features)  # [batch_size, embedding_dim]
        material_embed = self.material_embedding(material_features)  # [batch_size, embedding_dim]

        # 3. 拼接结构化特征作为目标序列
        tgt_features = torch.stack([size_embed, operation_embed, material_embed], dim=1)  # [batch_size, 3, embedding_dim]

        # 4. 图像特征作为记忆序列
        memory_features = image_features.unsqueeze(1)  # [batch_size, 1, embedding_dim]

        # 5. Transformer Decoder
        lifetime_pred = self.decoder(tgt_features, memory_features)

        return lifetime_pred

def create_lifetime_model(config, pretrained_damage_model_path=None, pretrained_backbone=True):
    """
    创建寿命预测模型

    Args:
        config: 配置字典
        pretrained_damage_model_path: 预训练损伤识别模型路径

    Returns:
        寿命预测模型
    """
    model = LifetimePredictionModel(config, pretrained_damage_model_path, pretrained_backbone)
    return model

if __name__ == "__main__":
    import json
    import os

    # 加载配置
    config_path = str(Path(__file__).resolve().parents[1] / 'configs/config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = create_lifetime_model(config)
    model = model.to(device)

    # 创建测试输入
    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_size = torch.randn(batch_size, 7).to(device)  # 尺寸特征：7维
    dummy_operation = torch.randn(batch_size, 2).to(device)  # 工况特征：2维
    dummy_material = torch.randn(batch_size, 8).to(device)  # 材料特征：8维

    with torch.no_grad():
        output = model(dummy_image, dummy_size, dummy_operation, dummy_material)

    print(f"输入图像形状: {dummy_image.shape}")
    print(f"输入尺寸特征形状: {dummy_size.shape}")
    print(f"输入工况特征形状: {dummy_operation.shape}")
    print(f"输入材料特征形状: {dummy_material.shape}")
    print(f"输出寿命预测形状: {output.shape}")
    print(f"预测寿命值: {output.squeeze().cpu().numpy()}")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"冻结参数量: {total_params - trainable_params:,}")
