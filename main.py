import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

# FastAPI application for plant disease prediction
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Smart Farming Plant Disease Prediction API!"}


# Allow requests from all origins (for Flutter frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained TensorFlow model
model = load_model("Smart_Farming_DL_Model.h5")

# Class names in the same order as training
class_names = [
    "Apple___scab", "Apple___Black_rot", "Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites",
    "Tomato___Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", "Tomato___healthy"
]

# Mapping of disease to best medicine
disease_to_medicine = {
    "Apple___scab": "Captan 50% WP spray",
    "Apple___Black_rot": "Copper-based fungicides",
    "Cedar_apple_rust": "Myclobutanil fungicide",
    "Apple___healthy": "No treatment needed",
    "Blueberry___healthy": "No treatment needed",
    "Cherry_(including_sour)___Powdery_mildew": "Sulfur-based fungicide",
    "Cherry_(including_sour)___healthy": "No treatment needed",
    "Corn_(maize)___Cercospora Gray_leaf_spot": "Azoxystrobin spray",
    "Corn_(maize)___Common_rust_": "Fungicide with tebuconazole",
    "Corn_(maize)___Northern_Leaf_Blight": "Trifloxystrobin spray",
    "Corn_(maize)___healthy": "No treatment needed",
    "Grape___Black_rot": "Mancozeb or Ziram",
    "Grape___Esca_(Black_Measles)": "Remove infected vines",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Copper fungicide spray",
    "Grape___healthy": "No treatment needed",
    "Orange___Haunglongbing_(Citrus_greening)": "No cure, remove infected trees",
    "Peach___Bacterial_spot": "Oxytetracycline spray",
    "Peach___healthy": "No treatment needed",
    "Pepper,_bell___Bacterial_spot": "Copper fungicide spray",
    "Pepper,_bell___healthy": "No treatment needed",
    "Potato___Early_blight": "Chlorothalonil fungicide",
    "Potato___Late_blight": "Mancozeb 75% WP every 7 days",
    "Potato___healthy": "No treatment needed",
    "Raspberry___healthy": "No treatment needed",
    "Soybean___healthy": "No treatment needed",
    "Squash___Powdery_mildew": "Neem oil or sulfur-based sprays",
    "Strawberry___Leaf_scorch": "Use certified disease-free plants",
    "Strawberry___healthy": "No treatment needed",
    "Tomato___Bacterial_spot": "Copper-based fungicide",
    "Tomato___Early_blight": "Chlorothalonil spray",
    "Tomato___Late_blight": "Mancozeb fungicide weekly",
    "Tomato___Leaf_Mold": "Copper spray or baking soda solution",
    "Tomato___Septoria_leaf_spot": "Fungicide like chlorothalonil",
    "Tomato___Spider_mites": "Insecticidal soap or neem oil",
    "Tomato___Target_Spot": "Azoxystrobin fungicide",
    "Tomato_Yellow_Leaf_Curl_Virus": "No cure, use resistant seeds",
    "Tomato_mosaic_virus": "Remove infected plants, clean tools",
    "Tomato___healthy": "No treatment needed"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        filename = file.filename.lower()
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            return JSONResponse(
                content={"error": "Only .jpg, .jpeg, and .png files are supported."},
                status_code=400
            )
        contents = await file.read()
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB").resize((128, 128))
        except Exception:
            return JSONResponse(
                content={"error": "Invalid or corrupted image file."},
                status_code=400
            )
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        confidence = float(np.max(prediction))
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]

        if confidence < 0.89:
            return JSONResponse(
                content={"error":" Please upload a valid plant leaf image."},
                status_code=400,
            )
        
        medicine = disease_to_medicine.get(predicted_class, "No recommendation found")

        return JSONResponse({
            "Disease": predicted_class,
            "Recommendation": medicine
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
