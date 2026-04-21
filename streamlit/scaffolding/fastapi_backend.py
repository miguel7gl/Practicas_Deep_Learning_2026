from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI(title="Chachi Pistachi Real Estate API")

# 1. Definir las clases EXACTAMENTE como salen en tu entrenamiento
CLASSES = [
    'Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial', 
    'Inside city', 'Kitchen', 'Living room', 'Mountain', 'Office', 
    'Open country', 'Store', 'Street', 'Suburb', 'Tall building'
]

# 2. Cargar el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    # Usamos la misma arquitectura: EfficientNet-B3
    model = models.efficientnet_b3()
    num_ftrs = model.classifier[1].in_features
    # Ajustamos a las 15 clases del proyecto
    model.classifier[1] = nn.Linear(num_ftrs, len(CLASSES)) 
    
    # Cargamos los pesos (asegúrate de que el nombre coincida con tu archivo descargado)
    model.load_state_dict(torch.load("model_colorful-sweep-3_acc_0.91.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# 3. Transformaciones de la imagen (deben ser iguales a las de entrenamiento)
preprocess = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer la imagen
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # Preprocesar
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Predecir
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, index = torch.max(probabilities, 0)
    
    return {
        "filename": file.filename,
        "label": CLASSES[index.item()],
        "confidence": float(confidence.item()),
        "status": "success"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)