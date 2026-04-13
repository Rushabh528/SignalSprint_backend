import os
import pickle
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import gdown

MODEL_PATH = "model.pkl"
FILE_ID = "1qElVvY1xryWN4J3YrLCt3lw7U88o-CM5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# download first
download_model()

# load model safely
print("Loading model...")
model = pickle.load(open(MODEL_PATH, "rb"))
print("Model loaded!")

app = FastAPI()


def preprocess(image):
    image = image.resize((224, 224))
    return np.array(image).reshape(1, -1)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    data = preprocess(image)
    pred = model.predict(data)
    return {"prediction": int(pred[0])}