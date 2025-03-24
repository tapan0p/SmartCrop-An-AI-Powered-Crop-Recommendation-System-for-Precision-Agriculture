from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch 
import torch.nn as nn 
import numpy as np 
from typing import List 
import joblib

app = FastAPI(title="SmartCrop API")

class CropInput(BaseModel):
    N : float
    P: float
    K: float 
    temperature: float 
    humidity: float 
    ph: float 
    rainfall: float

class CropPrediction(BaseModel):
    crop: str 
    confidence: float 

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_class):
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_dim, num_class, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(num_class, dtype=torch.float32))
        self.temparature = 0.1
        
    def forward(self, x):
        return (x @ self.W + self.b)/self.temparature
    
try:

    model = LogisticRegression(input_dim=7, num_class=22)
    model.load_state_dict(torch.load('..\Models\Logistic_regression_model.pth'))
    model.eval()
    print("Model loaded sucessfully")

    label_encoder = joblib.load('..\Label_Encoder\label_encoder.pkl')

except Exception as e:
    print(f"Error loading the model : {str(e)}")

@app.post('/predict')
async def predict_corp(data:CropInput):
    try:
        input_data = np.array([
            data.N,data.P,data.K,data.temperature,data.humidity,data.ph,data.rainfall
        ]).reshape(1,-1)
        input_tensor = torch.FloatTensor(input_data)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = nn.Softmax(dim=1)(logits)
            confidence, predicted = torch.max(probs, 1)

        crop_name = label_encoder.inverse_transform([predicted.item()])[0]
        print({
            "crop": crop_name,
            "confidence": round(confidence.item() * 100, 2),
            "probs":probs.tolist()
        })
        
        return {
            "crop": crop_name,
            "confidence": confidence.item()
        }
    
    except Exception as e:
        print(f"Error predicting : {str(e)}")

@app.get('/hi')
def home():
    return {"home":"hello"}