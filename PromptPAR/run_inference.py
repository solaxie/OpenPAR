import os
import torch
from torchvision import transforms
from PIL import Image
from clip import clip
from clip.model import *

# Define ATTRIBUTES
ATTRIBUTES = [
    "male", "hat", "glasses", "shirt", "long hair", 
    "shorts", "jeans", "long pants", "skirt", "dress", 
    "running", "walking", "standing", "sitting"
]

# Hardcoded paths
IMAGE_PATH = "/content/drive/MyDrive/PA-100K/vlcsnap-2024-09-28-20h04m06s278.jpg"
CHECKPOINT_PATH = "/content/drive/MyDrive/PA-100K/PA100k_Checkpoint.pth"

class TransformerClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(TransformerClassifier, self).__init__()
        self.clip_model = clip_model
        self.num_classes = num_classes
        self.classifier = nn.Linear(clip_model.visual.output_dim, num_classes)

    def forward(self, image):
        features = self.clip_model.encode_image(image)
        output = self.classifier(features)
        return output

def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    clip_model = build_model(checkpoint['ViT_model'])
    model = TransformerClassifier(clip_model, len(ATTRIBUTES))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.sigmoid(output)
    return probabilities.squeeze().cpu().numpy()

def run_inference():
    # Load the model
    model = load_model(CHECKPOINT_PATH)

    # Preprocess the image
    image_tensor = preprocess_image(IMAGE_PATH)

    # Make prediction
    probabilities = predict(model, image_tensor)

    # Print results
    for attr, prob in zip(ATTRIBUTES, probabilities):
        print(f"{attr}: {prob:.4f}")

if __name__ == '__main__':
    run_inference()