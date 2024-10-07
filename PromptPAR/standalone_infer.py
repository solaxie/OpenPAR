import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# CLIP ViT-L/14 模型的简化版本
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution=224, patch_size=14, width=1024, layers=24, heads=16, output_dim=768):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=width * 4),
            num_layers=layers
        )
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

class TransformerClassifier(nn.Module):
    def __init__(self, attr_num, dim=768):
        super().__init__()
        self.attr_num = attr_num
        self.vit = VisionTransformer()
        self.word_embed = nn.Linear(self.vit.output_dim, dim)
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for _ in range(self.attr_num)])
        self.dim = dim
        self.bn = nn.BatchNorm1d(self.attr_num)

    def forward(self, imgs):
        clip_image_features = self.vit(imgs)
        x = self.word_embed(clip_image_features).unsqueeze(1).repeat(1, self.attr_num, 1)
        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        bn_logits = self.bn(logits)
        return bn_logits

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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_state_dict = checkpoint['model_state_dict']
    attr_num = checkpoint.get('attr_num', 26)  # 默认值
    attributes = checkpoint.get('attributes', [f"Attribute_{i}" for i in range(attr_num)])
    
    model = TransformerClassifier(attr_num)
    
    # 只加载匹配的参数
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in model_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    model.eval()
    return model, attributes

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, attributes = load_model(args.checkpoint)
    model.to(device)

    image = load_image(args.image_path).to(device)

    with torch.no_grad():
        outputs = model(image)
    
    predictions = torch.sigmoid(outputs) > 0.5

    print("Detected attributes:")
    for attr, pred in zip(attributes, predictions[0]):
        if pred:
            print(f"- {attr}")

if __name__ == "__main__":
    main()