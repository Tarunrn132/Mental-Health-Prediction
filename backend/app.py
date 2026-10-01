import os
import json
import logging
from typing import Literal, Optional
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Setup logging - avoid logging sensitive user health details
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("mental-health-api")

TOP_COUNTRIES = [
    'Other', 'India', 'USA', 'Canada', 'Australia',
    'UK', 'Germany', 'Mexico', 'Turkey', 'France'
]

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "mental_health_pipeline.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# Load Model
model = None
metrics_data = {}

def load_pipeline():
    global model, metrics_data
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            logger.info("Successfully loaded mental health prediction pipeline.")
        except Exception as e:
            logger.error(f"Failed to load model pipeline: {e}")
            model = None
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}")

    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metrics: {e}")

load_pipeline()

app = FastAPI(
    title="Mental Health Signal API",
    description="Student Wellness Analytics - Lifestyle and Stress Habit Predictive Engine",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom validation error handler for human-readable responses
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field = " -> ".join([str(loc) for loc in err.get("loc", []) if loc != "body"])
        errors.append({
            "field": field,
            "message": err.get("msg", "Invalid value"),
            "type": err.get("type", "value_error")
        })
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Input validation failed. Please check the submitted fields.",
            "detail": exc.errors(),
            "field_errors": errors
        }
    )

# Pydantic Schemas
class StudentInput(BaseModel):
    age: int = Field(..., ge=10, le=100, description="Age in years (10-100)")
    gender: Literal['Male', 'Female'] = Field(..., description="Gender")
    country: str = Field(..., min_length=1, max_length=100, description="Country of residence")
    academic_level: Literal['High School', 'Undergraduate', 'Graduate'] = Field(
        ..., description="Current academic stage"
    )
    most_used_platform: Literal[
        'Facebook', 'Instagram', 'Snapchat', 'Twitter', 'YouTube',
        'TikTok', 'LinkedIn', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat'
    ] = Field(..., description="Primary social media platform")
    purpose_of_use: Literal['Networking', 'Education', 'Entertainment', 'News'] = Field(
        ..., description="Main purpose of digital usage"
    )
    avg_daily_usage_hours: float = Field(
        ..., ge=0.0, le=24.0, description="Average daily screen time (0-24 hrs)"
    )
    daily_unlocks: int = Field(
        ..., ge=0, description="Daily smartphone unlocks"
    )
    study_hours: float = Field(
        ..., ge=0.0, le=24.0, description="Daily study hours (0-24 hrs)"
    )
    physical_activity_hours: float = Field(
        ..., ge=0.0, le=24.0, description="Daily physical activity hours (0-24 hrs)"
    )
    sleep_hours_per_night: float = Field(
        ..., ge=0.0, le=24.0, description="Sleep duration per night (0-24 hrs)"
    )
    stress_level: Literal['Low', 'Medium', 'High', 'Very High'] = Field(
        ..., description="Perceived subjective stress level"
    )

class PredictionResponse(BaseModel):
    success: bool = True
    score: float
    predicted_mental_health_score: float
    band: str
    context: str
    scale_min: float = 0.0
    scale_max: float = 10.0
    disclaimer: str = (
        "This score is a machine-learning estimate based on habit patterns. "
        "It is not a clinical assessment or medical diagnosis."
    )

def interpret_score(score: float):
    if score < 4.0:
        return (
            "Signal: strained",
            "Your responses suggest elevated strain right now. Small shifts in sleep or screen time can go a long way."
        )
    elif score < 7.0:
        return (
            "Signal: balanced",
            "Your rhythm looks fairly steady, with some room to recover and reset."
        )
    else:
        return (
            "Signal: strong",
            "Your habits point to a well-supported, resilient baseline. Keep it up."
        )

from fastapi.responses import JSONResponse, FileResponse

# Routes
@app.get("/", include_in_schema=False)
def root(request: Request):
    accept = request.headers.get("accept", "")
    # If a browser requests HTML and not specifically JSON
    if "text/html" in accept and "application/json" not in accept and (FRONTEND_DIR / "index.html").exists():
        return FileResponse(FRONTEND_DIR / "index.html")
    return {
        "name": "Mental Health Signal — Student Wellness Analytics API",
        "status": "online",
        "model_loaded": model is not None,
        "disclaimer": "Informational wellness analytics engine; not a medical diagnosis tool."
    }

@app.get("/style.css", include_in_schema=False)
def serve_css():
    css_file = FRONTEND_DIR / "style.css"
    if css_file.exists():
        return FileResponse(css_file, media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")

@app.get("/script.js", include_in_schema=False)
def serve_js():
    js_file = FRONTEND_DIR / "script.js"
    if js_file.exists():
        return FileResponse(js_file, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JS file not found")

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_ready": model is not None,
        "metrics_summary": metrics_data.get("test_metrics", None)
    }

@app.get("/api/metrics")
def get_metrics():
    if not metrics_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation metrics are not yet available. Run train.py first."
        )
    return metrics_data

@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentInput):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The prediction model is currently not loaded. Please verify model file."
        )

    # Normalize country for model
    country_clean = data.country.strip()
    country_group = country_clean if country_clean in TOP_COUNTRIES else "Other"

    # Prepare DataFrame row matching model training columns
    input_row = pd.DataFrame([{
        'Study_Hours': float(data.study_hours),
        'Age': int(data.age),
        'Avg_Daily_Usage_Hours': float(data.avg_daily_usage_hours),
        'Daily_Unlocks': int(data.daily_unlocks),
        'Physical_Activity_Hours': float(data.physical_activity_hours),
        'Sleep_Hours_Per_Night': float(data.sleep_hours_per_night),
        'Stress_Level': data.stress_level,
        'Gender': data.gender,
        'Academic_Level': data.academic_level,
        'Most_Used_Platform': data.most_used_platform,
        'Purpose_Of_Use': data.purpose_of_use,
        'Grouped_country': country_group
    }])

    try:
        raw_pred = float(model.predict(input_row)[0])
        # Bound score between 0.0 and 10.0
        clamped_score = max(0.0, min(10.0, raw_pred))
        final_score = round(clamped_score, 2)
        band, context = interpret_score(final_score)

        logger.info(f"Processed prediction request successfully -> score: {final_score}")

        return PredictionResponse(
            success=True,
            score=final_score,
            predicted_mental_health_score=final_score,
            band=band,
            context=context
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the prediction score. Please try again."
        )

# Mount frontend files if existing
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
