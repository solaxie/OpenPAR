import argparse
import torch
from PIL import Image
from torchvision import transforms
from clip import clip
from models.base_block import TransformerClassifier

# 定义属性列表（与PA100k数据集相对应）
ATTRIBUTES = [
    'Female', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Front', 'Side', 'Back',
    'Hat', 'Glasses', 'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront',
    'ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice',
    'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress', 'boots'
]

def load_model(checkpoint_path):
    # 加载CLIP模型
    clip_model, _ = clip.load("ViT-L/14", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建TransformerClassifier模型
    model = TransformerClassifier(clip_model, len(ATTRIBUTES), ATTRIBUTES)
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, clip_model

def preprocess_image(image_path):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def infer(model, clip_model, image):
    model.eval()
    with torch.no_grad():
        logits, _ = model(image, clip_model)
        predictions = torch.sigmoid(logits) > 0.5
    return predictions

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    model, clip_model = load_model(args.checkpoint)
    model.to(device)
    clip_model.to(device)

    # 预处理图像
    image = preprocess_image(args.image).to(device)

    # 进行推理
    predictions = infer(model, clip_model, image)

    # 输出结果
    print("Detected attributes:")
    for attr, pred in zip(ATTRIBUTES, predictions[0]):
        if pred:
            print(f"- {attr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()

    main(args)