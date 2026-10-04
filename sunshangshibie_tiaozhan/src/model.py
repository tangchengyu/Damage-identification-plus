import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class VGGDamageClassifier(nn.Module):
    """基于VGG的齿轮损伤分类器基线模型"""

    def __init__(self, num_classes=4, pretrained=True, model_name='vgg16'):
        """
        初始化VGG模型

        Args:
            num_classes: 分类类别数，默认4类
            pretrained: 是否使用预训练权重
            model_name: 模型名称 ('vgg11', 'vgg13', 'vgg16', 'vgg19')
        """
        super(VGGDamageClassifier, self).__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        # 根据模型名称创建不同的VGG模型
        weights = 'IMAGENET1K_V1' if pretrained else None

        if model_name == 'vgg11':
            self.backbone = models.vgg11(weights=weights)
        elif model_name == 'vgg13':
            self.backbone = models.vgg13(weights=weights)
        elif model_name == 'vgg16':
            self.backbone = models.vgg16(weights=weights)
        elif model_name == 'vgg19':
            self.backbone = models.vgg19(weights=weights)
        else:
            raise ValueError(f"不支持的VGG模型: {model_name}")

        # 替换最后的分类器
        self.backbone.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        """前向传播"""
        return self.backbone(x)

    def get_features(self, x):
        """获取特征表示（不包括最后的分类层）"""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def create_vgg_damage_classifier(num_classes=4, pretrained=True, model_name='vgg16'):
    """
    创建基于VGG的齿轮损伤分类器

    Args:
        num_classes: 分类类别数，默认4类
        pretrained: 是否使用预训练权重
        model_name: VGG模型名称

    Returns:
        VGGDamageClassifier: 初始化的VGG模型
    """
    model = VGGDamageClassifier(num_classes=num_classes, pretrained=pretrained, model_name=model_name)
    return model

class DamageClassifier(nn.Module):
    """齿轮损伤分类器模型"""

    def __init__(self, num_classes=4, pretrained=True, model_name='resnet34'):
        """
        初始化模型

        Args:
            num_classes: 分类类别数，默认4类
            pretrained: 是否使用预训练权重
            model_name: 模型名称 ('resnet34', 'resnet50', 'resnet18')
        """
        super(DamageClassifier, self).__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        # 根据模型名称创建不同的ResNet模型
        # 使用weights参数替代已弃用的pretrained参数
        weights = 'IMAGENET1K_V1' if pretrained else None

        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights=weights)
            num_ftrs = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(weights=weights)
            num_ftrs = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(weights=weights)
            num_ftrs = 2048
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        # 替换最后的全连接层
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """前向传播"""
        return self.backbone(x)

    def get_features(self, x):
        """获取特征表示（不包括最后的分类层）"""
        # 提取除了最后全连接层的所有特征
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return x

def create_damage_classifier(num_classes=4, pretrained=True, model_name='resnet34'):
    """
    创建齿轮损伤分类器

    Args:
        num_classes: 分类类别数，默认4类
        pretrained: 是否使用预训练权重
        model_name: 模型名称

    Returns:
        DamageClassifier: 初始化的模型
    """
    model = DamageClassifier(num_classes=num_classes, pretrained=pretrained, model_name=model_name)
    return model

class MultiModalDamageClassifier(nn.Module):
    """多模态齿轮损伤分类器（图像+技术参数）"""

    def __init__(self, num_classes=4, pretrained=True, technical_params_dim=15):
        """
        初始化多模态模型

        Args:
            num_classes: 分类类别数
            pretrained: 是否使用预训练权重
            technical_params_dim: 技术参数维度
        """
        super(MultiModalDamageClassifier, self).__init__()

        # 图像分支 - ResNet34
        # 使用weights参数替代已弃用的pretrained参数
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.image_backbone = models.resnet34(weights=weights)
        self.image_backbone.fc = nn.Identity()  # 移除最后的分类层

        # 技术参数分支 - MLP
        self.technical_mlp = nn.Sequential(
            nn.Linear(technical_params_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(512 + 64, 256),  # 512 (image) + 64 (technical)
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, technical_params=None):
        """
        前向传播

        Args:
            image: 输入图像
            technical_params: 技术参数，可选
        """
        # 图像特征提取
        image_features = self.image_backbone(image)

        if technical_params is not None:
            # 技术参数特征提取
            technical_features = self.technical_mlp(technical_params)

            # 特征融合
            fused_features = torch.cat([image_features, technical_features], dim=1)
        else:
            # 只使用图像特征
            fused_features = torch.cat([image_features, torch.zeros(image_features.size(0), 64).to(image_features.device)], dim=1)

        # 最终分类
        output = self.fusion_layer(fused_features)
        return output

def initialize_weights(model):
    """初始化模型权重"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def load_model(model_path, num_classes=4, model_name='resnet34'):
    """
    从文件加载模型

    Args:
        model_path: 模型文件路径
        num_classes: 分类类别数
        model_name: 模型名称

    Returns:
        model: 加载的模型
    """
    model = create_damage_classifier(num_classes=num_classes, pretrained=False, model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model

def save_model(model, model_path):
    """
    保存模型到文件

    Args:
        model: 要保存的模型
        model_path: 保存路径
    """
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 测试VGG基线模型
    print("创建VGG基线模型:")
    vgg_model = create_vgg_damage_classifier(num_classes=4, pretrained=True, model_name='vgg16')
    vgg_model = vgg_model.to(device)

    # 测试前向传播
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    vgg_output = vgg_model(dummy_input)
    print(f"VGG输入形状: {dummy_input.shape}")
    print(f"VGG输出形状: {vgg_output.shape}")

    # 打印VGG参数量
    vgg_total_params = sum(p.numel() for p in vgg_model.parameters())
    vgg_trainable_params = sum(p.numel() for p in vgg_model.parameters() if p.requires_grad)
    print(f"VGG总参数量: {vgg_total_params:,}")
    print(f"VGG可训练参数量: {vgg_trainable_params:,}")

    # 测试ResNet模型
    print("\n创建ResNet模型:")
    resnet_model = create_damage_classifier(num_classes=4, pretrained=True, model_name='resnet34')
    resnet_model = resnet_model.to(device)

    resnet_output = resnet_model(dummy_input)
    print(f"ResNet输入形状: {dummy_input.shape}")
    print(f"ResNet输出形状: {resnet_output.shape}")

    # 打印ResNet参数量
    resnet_total_params = sum(p.numel() for p in resnet_model.parameters())
    resnet_trainable_params = sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)
    print(f"ResNet总参数量: {resnet_total_params:,}")
    print(f"ResNet可训练参数量: {resnet_trainable_params:,}")

    # 测试多模态模型
    print("\n创建多模态模型:")
    multimodal_model = MultiModalDamageClassifier(num_classes=4, pretrained=True)
    multimodal_model = multimodal_model.to(device)

    dummy_technical = torch.randn(2, 15).to(device)
    multimodal_output = multimodal_model(dummy_input, dummy_technical)
    print(f"多模态输出形状: {multimodal_output.shape}")

    multimodal_total_params = sum(p.numel() for p in multimodal_model.parameters())
    multimodal_trainable_params = sum(p.numel() for p in multimodal_model.parameters() if p.requires_grad)
    print(f"多模态总参数量: {multimodal_total_params:,}")
    print(f"多模态可训练参数量: {multimodal_trainable_params:,}")
