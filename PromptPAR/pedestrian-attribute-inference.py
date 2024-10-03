import argparse
import torch
from PIL import Image
from torchvision import transforms
from clip import clip
from clip.model import build_model

class TransformerClassifier(torch.nn.Module):
    def __init__(self, clip_model, num_attributes):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = torch.nn.Linear(clip_model.visual.output_dim, num_attributes)

    def forward(self, image):
        features = self.clip_model.encode_image(image)
        return self.classifier(features)

def load_model(checkpoint_path, num_attributes):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    clip_model = build_model(checkpoint['ViT_model'])
    model = TransformerClassifier(clip_model, num_attributes)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return model, clip_model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def infer(model, clip_model, image_tensor, device):
    model.eval()
    clip_model.eval()
    with torch.no_grad():
        clip_features = clip_model.encode_image(image_tensor)
        outputs = model(clip_features)
    return torch.sigmoid(outputs).squeeze().cpu().numpy()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 這裡的屬性列表應該與您的模型相匹配
    attributes = [
        "male", "long hair", "sunglasses", "hat", "t-shirt", "long sleeves",
        "formal", "short", "jeans", "long pants", "skirt", "face visible",
        "arms behind back"
    ]
    num_attributes = len(attributes)

    model, clip_model = load_model(args.checkpoint_path, num_attributes)
    model.to(device)
    clip_model.to(device)

    image_tensor = preprocess_image(args.image_path).to(device)
    probabilities = infer(model, clip_model, image_tensor, device)

    print("Pedestrian Attribute Probabilities:")
    for attr, prob in zip(attributes, probabilities):
        print(f"{attr}: {prob:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pedestrian Attribute Recognition")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    main(args)
