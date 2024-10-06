import sys
import torch
from PIL import Image
from torchvision import transforms
from models.base_block import TransformerClassifier
from clip import clip
from clip.model import build_model

class DummyArgs:
    def __init__(self, dataset):
        self.dataset = dataset
        self.use_div = False
        self.use_vismask = False
        self.use_GL = False
        self.use_textprompt = False
        self.use_mm_former = False
        self.mm_layers = 1
        self.div_num = 4
        self.overlap_row = 2
        self.text_prompt = 3
        self.vis_prompt = 50
        self.vis_depth = 24

def load_model(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        dummy_args = DummyArgs('PA100k')  # Adjust this if needed
        clip_model = build_model(checkpoint['ViT_model'], dummy_args)
        attributes = checkpoint.get('attributes', [])
        attr_num = len(attributes)
        model = TransformerClassifier(clip_model, attr_num, attributes, dummy_args)
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

def main(checkpoint_path, image_path):
    print(f"Running inference with checkpoint: {checkpoint_path}")
    print(f"Image path: {image_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model, clip_model, attributes = load_model(checkpoint_path)
        model.to(device)
        clip_model.to(device)

        image_tensor = preprocess_image(image_path).to(device)
        probabilities = run_inference(model, clip_model, image_tensor)

        for attr, prob in zip(attributes, probabilities):
            print(f"{attr}: {prob:.4f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python standalone_infer.py <checkpoint_path> <image_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    image_path = sys.argv[2]
    main(checkpoint_path, image_path)