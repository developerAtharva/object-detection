import base64
import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO
from fastapi.responses import JSONResponse


import os
import requests

MODEL_URL = "https://huggingface.co/developerAtharva/YOLO/resolve/main/best_new.pt"
MODEL_PATH = "models/best_new.pt"

# Ensure models/ directory exists
os.makedirs("models", exist_ok=True)

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Model downloaded.")


# Load YOLO model once
model = YOLO(MODEL_PATH)
class_names = model.names

app = FastAPI()

class ImageRequest(BaseModel):
    image_base64: str

@app.post("/predict")
async def predict(request: ImageRequest):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run prediction
        results = model.predict(img, conf=0.25, line_width=1)

        predictions = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0].tolist())
                class_name = class_names[class_id]
                coords = box.xyxy[0].tolist()
                predictions.append({
                    "class": class_name,
                    "coordinates": {
                        "x1": coords[0],
                        "y1": coords[1],
                        "x2": coords[2],
                        "y2": coords[3]
                    }
                })

        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
