import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from timm.models.vision_transformer import VisionTransformer

def load_trained_model(model_path):
    # Instantiate the model architecture
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, num_classes=80000)
    
    # Load the state dictionary from the checkpoint
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model_weights = fix_state_dict_mismatches(state_dict["model"])
    
    # Load the modified state_dict into the model
    model.load_state_dict(model_weights, strict=False)
    model.eval()
    
    return model

def fix_state_dict_mismatches(state_dict):
    # Rename mismatched keys
    if "fc_norm.weight" in state_dict and "fc_norm.bias" in state_dict:
        state_dict["norm.weight"] = state_dict.pop("fc_norm.weight")
        state_dict["norm.bias"] = state_dict.pop("fc_norm.bias")
    
    # Remove size-mismatched keys (assuming you don't need the classification head)
    if "head.weight" in state_dict and "head.bias" in state_dict:
        del state_dict["head.weight"]
        del state_dict["head.bias"]
    
    return state_dict

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path)
    return transform(img).unsqueeze(0)

def predict_image(image_path, model_path):
    model = load_trained_model(model_path)
    img = preprocess_image(image_path)
    
    with torch.no_grad():
        output = model(img)
        
    return torch.argmax(output, dim=1).item()

model_path = "PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth"
image_path = "IMG_2645.jpg"
prediction = predict_image(image_path, model_path)
print(prediction)
