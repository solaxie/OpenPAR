import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import math
from collections import OrderedDict

# CLIP模型相关类和函数
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))[0]
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

class CLIP(nn.Module):
    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: int, vision_width: int, vision_patch_size: int):
        super().__init__()
        self.output_dim = embed_dim
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim
        )

    def encode_image(self, image):
        return self.visual(image.type(self.visual.conv1.weight.dtype))

# TransformerClassifier 类
class TransformerClassifier(nn.Module):
    def __init__(self, clip_model, attr_num, dim=768):
        super().__init__()
        self.attr_num = attr_num
        self.word_embed = nn.Linear(clip_model.output_dim, dim)
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for _ in range(self.attr_num)])
        self.dim = dim
        self.bn = nn.BatchNorm1d(self.attr_num)

    def forward(self, imgs, clip_model):
        clip_image_features = clip_model.encode_image(imgs)
        x = self.word_embed(clip_image_features).unsqueeze(1).repeat(1, self.attr_num, 1)
        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        bn_logits = self.bn(logits)
        return bn_logits, None

# 辅助函数
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def load_model(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 提取必要信息
        model_state_dict = checkpoint['model_state_dict']
        attr_num = checkpoint.get('attr_num', 26)  # 默认值
        attributes = checkpoint.get('attributes', [f"Attribute_{i}" for i in range(attr_num)])  # 默认属性列表
        
        # 初始化CLIP模型
        clip_model = CLIP(
            embed_dim=512,
            image_resolution=224,
            vision_layers=24,
            vision_width=1024,
            vision_patch_size=14
        )
        
        print(f"CLIP model structure: {clip_model}")  # 调试输出
        
        # 初始化并加载模型
        model = TransformerClassifier(clip_model, attr_num)
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()
        
        return model, clip_model, attributes
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model, clip_model, attributes = load_model(args.checkpoint)
    model.to(device)
    clip_model.to(device)

    # 加载并预处理图像
    image = load_image(args.image_path).to(device)

    # 推理
    with torch.no_grad():
        outputs, _ = model(image, clip_model)
    
    # 处理输出
    predictions = torch.sigmoid(outputs) > 0.5

    # 打印结果
    print("Detected attributes:")
    for attr, pred in zip(attributes, predictions[0]):
        if pred:
            print(f"- {attr}")

if __name__ == "__main__":
    main()