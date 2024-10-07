import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution=224, patch_size=16, width=768, layers=12, heads=12, output_dim=768):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, (input_resolution // patch_size) ** 2 + 1, width))
        self.blocks = nn.ModuleList([
            Block(width, heads) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(width)
        self.fc = nn.Linear(width, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([self.cls_token.repeat(x.shape[0], 1, 1), x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.fc(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, attr_num, dim=768):
        super().__init__()
        self.word_embed = nn.Linear(dim, dim)
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for _ in range(attr_num)])
        self.bn = nn.BatchNorm1d(attr_num)

    def forward(self, x):
        x = self.word_embed(x)
        logits = torch.cat([layer(x) for layer in self.weight_layer], dim=1)
        return self.bn(logits)

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    vit = VisionTransformer()
    vit.load_state_dict(checkpoint['ViT_model'], strict=False)
    
    classifier = TransformerClassifier(attr_num=26)  # PA100k has 26 attributes
    classifier.load_state_dict(checkpoint['model_state_dict'])
    
    return vit, classifier

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vit, classifier = load_model(args.checkpoint)
    vit.to(device)
    classifier.to(device)
    vit.eval()
    classifier.eval()

    image = load_image(args.image_path).to(device)

    with torch.no_grad():
        features = vit(image)
        outputs = classifier(features)
    
    predictions = torch.sigmoid(outputs) > 0.5

    # Assuming you have a list of attribute names
    attributes = [f"Attribute_{i}" for i in range(26)]  # Replace with actual attribute names
    
    print("Detected attributes:")
    for attr, pred in zip(attributes, predictions[0]):
        if pred:
            print(f"- {attr}")

if __name__ == "__main__":
    main()