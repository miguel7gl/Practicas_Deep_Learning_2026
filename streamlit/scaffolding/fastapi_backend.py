from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

app = FastAPI(title="Chachi Pistachi Real Estate API - Pro Version")

# 1. Definición de clases
CLASSES = [
    'Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial', 
    'Inside city', 'Kitchen', 'Living room', 'Mountain', 'Office', 
    'Open country', 'Store', 'Street', 'Suburb', 'Tall building'
]

# Configuración global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_model_name = "model_colorful-sweep-3_acc_0.91.pth" # Modelo por defecto
model = None

# 2. Lógica para cargar/recargar el modelo
def load_model(model_path: str):
    global model, current_model_name
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encuentra el archivo: {model_path}")
    
    # Arquitectura EfficientNet-B3
    new_model = models.efficientnet_b3()
    num_ftrs = new_model.classifier[1].in_features
    new_model.classifier[1] = nn.Linear(num_ftrs, len(CLASSES)) 
    
    # Cargar pesos
    new_model.load_state_dict(torch.load(model_path, map_location=device))
    new_model.to(device)
    new_model.eval()
    
    model = new_model
    current_model_name = model_path
    print(f"✅ Motor actualizado con: {model_path}")

# Carga inicial al arrancar la API
try:
    load_model(current_model_name)
except Exception as e:
    print(f"⚠️ Error carga inicial: {e}. Asegúrate de tener el archivo .pth en la carpeta.")

# 3. Preprocesamiento
preprocess = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- ENDPOINTS ---

@app.get("/current_model")
async def get_current_model():
    return {"current_model": current_model_name}

@app.post("/update_model")
async def update_model(model_filename: str):
    """Permite cambiar el modelo .pth que usa la API sin reiniciar"""
    try:
        load_model(model_filename)
        return {"status": "success", "updated_to": model_filename}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado en el servidor")
    
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, index = torch.max(probabilities, 0)
    
    return {
        "filename": file.filename,
        "label": CLASSES[index.item()],
        "confidence": float(confidence.item()),
        "model_used": current_model_name
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)