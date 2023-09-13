# Import necessary libraries
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image

# VisionTransformer model definition (based on models_vit.py)
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward_features(self, x):
        x = super().forward_features(x)
        if self.norm:
            x = self.norm(x)
        return x

# Load the trained model
model = VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, num_classes=80000)
# Extract model weights from the saved state dictionary
state_dict = torch.load("PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth", map_location=torch.device('cpu'))
model_weights = state_dict["model"]

# Load the model weights
model.load_state_dict(model_weights)
model.eval()

# Preprocess the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = "your_image_path.jpg"  # Replace with your image path
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0)

# Run inference
with torch.no_grad():
    predictions = model(input_tensor)

# Interpret the model's output (assuming classification)
predicted_class = predictions.argmax(dim=1).item()

print(f"Predicted class for image {image_path}: {predicted_class}")
