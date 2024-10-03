import torch
from torch import nn
import numpy as np
from torchvision import transforms
from PIL import Image
from models.base_block import TransformerClassifier
from clip.model import build_model

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 載入checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 建立CLIP模型
    print("Building CLIP model...")
    clip_model = build_model(checkpoint['ViT_model'])

    # 建立並載入TransformerClassifier模型
    print("Building and loading TransformerClassifier...")
    model = TransformerClassifier(clip_model, args.attr_num, args.attributes)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # 將模型移至適當的設備
    model = model.to(device)
    clip_model = clip_model.to(device)
    model.eval()

    # 載入並處理圖像
    print("Loading and processing image...")
    image = load_image(args.image_path).to(device)

    # 執行推理
    print("Performing inference...")
    with torch.no_grad():
        output = model(image)
        probabilities = torch.sigmoid(output)

    # 輸出結果
    print("Attribute Recognition Results:")
    for attr, prob in zip(args.attributes, probabilities[0]):
        if prob > 0.5:
            print(f"{attr}: Yes (Probability: {prob.item():.2f})")
        else:
            print(f"{attr}: No (Probability: {prob.item():.2f})")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--attr_num', type=int, required=True, help='Number of attributes')
    parser.add_argument('--attributes', nargs='+', required=True, help='List of attribute names')
    args = parser.parse_args()
    main(args)