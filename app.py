import sys
import os
import certifi
import pandas as pd

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from uvicorn import run as app_run

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.url_feature_extraction import URLFeatureExtractor

# -------------------- ENV + DB --------------------
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")

# Initialize MongoDB safely
try:
    import pymongo
    ca = certifi.where()
    client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca, serverSelectionTimeoutMS=2000)
    # Test connection
    client.admin.command('ping')
    
    from networksecurity.constant.training_pipeline import (
        DATA_INGESTION_COLLECTION_NAME,
        DATA_INGESTION_DATABASE_NAME
    )
    
    database = client[DATA_INGESTION_DATABASE_NAME]
    collection = database[DATA_INGESTION_COLLECTION_NAME]
    print("✅ MongoDB connection successful")
except Exception as e:
    print(f"⚠️  MongoDB connection failed (non-critical): {e}")
    client = None
    database = None
    collection = None

# -------------------- FASTAPI --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")

# Initialize URL extractor
try:
    url_extractor = URLFeatureExtractor()
    print("✅ URLFeatureExtractor initialized successfully")
except Exception as e:
    print(f"❌ Warning: URLFeatureExtractor failed to initialize: {e}")
    url_extractor = None

print("\n📋 Routes registered:")
print("  - GET  /")
print("  - GET  /health")
print("  - GET  /train")
print("  - GET  /train-run")
print("  - POST /predict")
print("  - POST /predict-url")
print("  - GET  /url-result")
print("\n🚀 FastAPI app ready!\n")

# -------------------- ROUTES --------------------

# ✅ HEALTH CHECK
@app.get("/health")
def health_check():
    return {"status": "ok", "url_extractor": url_extractor is not None}

# ✅ HOME PAGE (your UI)
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ✅ TRAIN ROUTE
#from fastapi.responses import RedirectResponse
@app.get("/train")
def train_page(request: Request):
    return templates.TemplateResponse("train_status.html", {"request": request})
@app.get("/train-run")
def train_run():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return "done"

    except Exception as e:
        return f"Error: {str(e)}"


# ✅ PREDICT ROUTE
@app.post("/predict")
def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # Read CSV
        df = pd.read_csv(file.file)

        # Load model
        model = load_object("final_model/model.pkl")

        # Predict
        y_pred = model.predict(df)

        # Convert output (optional improvement)
        df["predicted_column"] = y_pred
        df["predicted_column"] = df["predicted_column"].replace({
            1: "Legitimate",
            0: "Phishing"
        })

        # Save output
        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)

        # Convert to HTML
        table_html = df.to_html(classes="table table-striped")

        return templates.TemplateResponse(
            "table.html",
            {"request": request, "table": table_html}
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)


# ✅ PREDICT URL ROUTE - Simple test version
@app.post("/predict-url")
async def predict_url_post(request: Request):
    """
    Predict phishing for a single URL via JSON POST
    Expects JSON with 'url' field
    """
    try:
        print(f"📨 Received prediction request")
        data = await request.json()
        url = data.get("url", "").strip()
        print(f"   URL: {url}")
        
        if not url:
            print(f"   ❌ Empty URL provided")
            return JSONResponse(
                {
                    "success": False,
                    "error": "URL cannot be empty"
                },
                status_code=400
            )
        
        if url_extractor is None:
            print(f"   ❌ URL extractor not initialized")
            return JSONResponse(
                {
                    "success": False,
                    "error": "URL extractor not available - system initialization failed"
                },
                status_code=500
            )
        
        # Extract features from URL
        print(f"   🔍 Extracting features...")
        try:
            features_df = url_extractor.extract_features_dataframe(url)
            print(f"   ✅ Features extracted: {features_df.shape}")
            # Also calculate phishing heuristic score
            phishing_score = url_extractor.calculate_phishing_score(url)
            print(f"   📊 Phishing heuristic score: {phishing_score:.1f}/100")
        except Exception as extract_error:
            print(f"   ❌ Feature extraction error: {extract_error}")
            return JSONResponse(
                {
                    "success": False,
                    "error": f"Feature extraction failed: {str(extract_error)}"
                },
                status_code=400
            )
        
        # Load model
        print(f"   📦 Loading model...")
        try:
            model = load_object("final_model/model.pkl")
            print(f"   ✅ Model loaded")
        except Exception as model_error:
            print(f"   ❌ Model loading error: {model_error}")
            return JSONResponse(
                {
                    "success": False,
                    "error": f"Model loading failed: {str(model_error)}"
                },
                status_code=500
            )
        
        # Make prediction
        print(f"   🎯 Making prediction...")
        try:
            y_pred = model.predict(features_df)
            
            # Get prediction probabilities if available
            try:
                y_proba = model.predict_proba(features_df)
                confidence = float(max(y_proba[0])) * 100
            except:
                confidence = 100.0
            
            # Convert prediction
            prediction_label = "Legitimate" if y_pred[0] == 1 else "Phishing"
            print(f"   ✅ Prediction: {prediction_label} ({confidence:.2f}%)")
            
            return JSONResponse(
                {
                    "success": True,
                    "url": url,
                    "prediction": prediction_label,
                    "confidence": round(confidence, 2),
                    "phishing_score": round(phishing_score, 1)
                }
            )
        except Exception as pred_error:
            print(f"   ❌ Prediction error: {pred_error}")
            return JSONResponse(
                {
                    "success": False,
                    "error": f"Prediction failed: {str(pred_error)}"
                },
                status_code=500
            )
        
    except Exception as e:
        print(f"   ❌ Request error: {e}")
        return JSONResponse(
            {
                "success": False,
                "error": f"Request error: {str(e)}"
            },
            status_code=500
        )


# ✅ URL RESULT PAGE
@app.get("/url-result")
def url_result(request: Request, url: str = "", prediction: str = "", confidence: float = 0, phishing_score: float = 0, error: str = None):
    """Display URL prediction results"""
    return templates.TemplateResponse(
        "url_result.html",
        {
            "request": request,
            "url": url,
            "prediction": prediction,
            "confidence": confidence,
            "phishing_score": phishing_score,
            "error": error
        }
    )


# -------------------- RUN --------------------
if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)