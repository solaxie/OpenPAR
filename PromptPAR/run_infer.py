import argparse
import torch
from PIL import Image
from torchvision import transforms
from models.base_block import TransformerClassifier
from clip import clip
from clip.model import build_model
import sys

def load_model(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        clip_model = build_model(checkpoint['ViT_model'])
        attributes = checkpoint.get('attributes', [])
        attr_num = len(attributes)
        model = TransformerClassifier(clip_model, attr_num, attributes)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return model, clip_model, attributes
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

def preprocess_image(image_path):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            return transform(img).unsqueeze(0)
    except Exception as e:
        raise RuntimeError(f"Error preprocessing image: {str(e)}")

@torch.no_grad()
def run_inference(model, clip_model, image_tensor):
    model.eval()
    clip_model.eval()
    try:
        logits, _ = model(image_tensor, clip_model=clip_model)
        return torch.sigmoid(logits).squeeze().cpu().numpy()
    except Exception as e:
        raise RuntimeError(f"Error during inference: {str(e)}")

def main(args):
    print(f"Running inference with checkpoint: {args.checkpoint}")
    print(f"Image path: {args.image_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load model and attributes
        model, clip_model, attributes = load_model(args.checkpoint)
        model.to(device)
        clip_model.to(device)

        # Preprocess image
        image_tensor = preprocess_image(args.image_path).to(device)

        # Run inference
        probabilities = run_inference(model, clip_model, image_tensor)

        # Print results
        for attr, prob in zip(attributes, probabilities):
            print(f"{attr}: {prob:.4f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    print("Argument list:", sys.argv)
    parser = argparse.ArgumentParser(description="Run inference on a single image using PromptPAR")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    main(args)