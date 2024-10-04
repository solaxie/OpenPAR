import os
import argparse
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from config import argument_parser
from models.base_block import TransformerClassifier
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from clip import clip
from clip.model import build_model

def load_external_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the dataset to get attributes (we only need this for attribute names)
    train_set = MultiModalAttrDataset(args=args, split=args.train_split, transform=None)
    
    # Load the model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    clip_model = build_model(checkpoint['ViT_model'])
    model = TransformerClassifier(clip_model, train_set.attr_num, train_set.attributes)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    clip_model = clip_model.to(device)

    # Set models to evaluation mode
    model.eval()
    clip_model.eval()

    # Get the transform
    _, valid_tsfm = get_transform(args)

    # Load and transform the external image
    img = load_external_image(args.image_path, valid_tsfm)
    img = img.to(device)

    # Perform inference
    with torch.no_grad():
        logits, _ = model(img, clip_model=clip_model)
        probs = torch.sigmoid(logits)

    # Process and print results
    attributes = train_set.attributes
    for attr, prob in zip(attributes, probs[0]):
        print(f"{attr}: {prob.item():.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference for external image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    
    # Add any other necessary arguments from the original config
    parser.add_argument("--dataset", type=str, default="PETA")
    parser.add_argument("--use_div", action='store_true')
    parser.add_argument("--use_vismask", action='store_true')
    parser.add_argument("--use_GL", action='store_true')
    parser.add_argument("--use_textprompt", action='store_true')
    parser.add_argument("--use_mm_former", action='store_true')
    parser.add_argument("--vis_prompt", type=int, default=50)
    
    args = parser.parse_args()
    main(args)