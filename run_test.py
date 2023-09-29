import torch
from PIL import Image
from torchvision import transforms
import models_vit  # Ensure this import is correct

def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def load_model(model_path):
    model = models_vit.vit_large_patch16(num_classes=80000)
    checkpoint = torch.load(model_path, map_location='cpu')
    adjusted_state_dict = {k.replace('fc_norm', 'norm'): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(adjusted_state_dict)
    model.eval()
    return model

model = load_model('PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth')

def predict(image_path, model):
    image = process_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)  # Get the class with the highest score
    return predicted.item()

image_path = 'IMG_2645.jpg'  # Replace with the path to your image
predicted_class = predict(image_path, model)
print(f'Predicted class: {predicted_class}')

