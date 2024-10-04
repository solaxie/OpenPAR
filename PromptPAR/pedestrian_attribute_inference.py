import os
import sys
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

def main():
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
    # Remove all command line arguments
    sys.argv = [sys.argv[0]]
    main()