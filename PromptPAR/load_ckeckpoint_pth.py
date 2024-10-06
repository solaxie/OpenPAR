import torch
from models.base_block import TransformerClassifier
from clip import clip

def load_model(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 提取必要信息
        model_state_dict = checkpoint['model_state_dict']
        attr_num = checkpoint.get('attr_num', 26)  # 默认值
        attributes = checkpoint.get('attributes', [...])  # 默认属性列表
        
        # 加载 CLIP 模型
        clip_model, _ = clip.load("ViT-L/14", device=device)
        
        # 初始化并加载模型
        model = TransformerClassifier(clip_model, attr_num, attributes)
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()  # 设置为评估模式
        
        return model, clip_model, attributes
    
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}")
        return None, None, None

# 使用示例
model, clip_model, attributes = load_model('/content/PA100k_Checkpoint.pth')
if model is not None:
    print("Model loaded successfully")
    print(f"Number of attributes: {len(attributes)}")
    print(f"Attributes: {attributes}")
else:
    print("Failed to load model")