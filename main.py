import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

# 1. Initialize FastAPI app ONLY ONCE
app = FastAPI()

# 2. Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mhs-mental-health-score.onrender.com",  # Your frontend domain
        "*"  # Or wildcard to allow all origins
    ],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load ML Model safely
try:
    model = joblib.load("Mental_Health_Model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

top_countries = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']


# 4. Pydantic Schemas
class StudentData(BaseModel):
    age                     : int = Field(..., ge=10, le=100)
    gender                  : Literal['Male', 'Female']
    country                 : str
    academic_level          : Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform      : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter', 'YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat']
    purpose_of_use          : Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours   : float = Field(..., ge=0, le=24)
    daily_unlocks           : int   = Field(..., ge=0)
    study_hours             : float = Field(..., ge=0, le=24)
    physical_activity_hours : float = Field(..., ge=0, le=24)
    sleep_hours_per_night   : float = Field(..., ge=0, le=24)
    stress_level            : Literal['Medium', 'Low', 'Very High', 'High']


class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


# 5. Endpoints
@app.get('/')
def greet():
    return {"message": "Welcome to my project"}


@app.post('/predict', response_model=PredictionResponse)
def predict(data: StudentData):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model is not loaded. Ensure Mental_Health_Model.pkl was saved with a compatible Python version."
        )

    country_group = data.country if data.country in top_countries else "Other"

    input_row = pd.DataFrame([{
        'Age'                   : data.age,
        'Gender'                : data.gender,
        'Country'               : data.country,
        'Academic_Level'        : data.academic_level,
        'Most_Used_Platform'    : data.most_used_platform,
        'Purpose_Of_Use'        : data.purpose_of_use,
        'Avg_Daily_Usage_Hours' : data.avg_daily_usage_hours,
        'Daily_Unlocks'         : data.daily_unlocks,
        'Study_Hours'           : data.study_hours,
        'Physical_Activity_Hours': data.physical_activity_hours,
        'Sleep_Hours_Per_Night' : data.sleep_hours_per_night,
        'Stress_Level'          : data.stress_level,
        'Grouped_country'       : country_group
    }])

    prediction = model.predict(input_row)[0]
    return PredictionResponse(predicted_mental_health_score=round(float(prediction), 2))