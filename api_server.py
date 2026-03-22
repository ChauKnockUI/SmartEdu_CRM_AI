import os
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -------------------------------------------------------------
# Configuration & Logging
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SmartEdu_CRM AI Service",
    description="Microservice providing AI predictions for Lead Scoring and Churn Prediction.",
    version="1.0.0"
)

# Global model variables
LEAD_SCORING_MODEL = None
CHURN_PREDICTION_MODEL = None

# -------------------------------------------------------------
# Application Startup Event
# -------------------------------------------------------------
@app.on_event("startup")
def load_ai_models():
    """
    Loads machine learning models into memory when the server starts.
    """
    global LEAD_SCORING_MODEL
    global CHURN_PREDICTION_MODEL
    
    # 1. Load Lead Scoring Model
    lead_model_path = os.path.join(os.path.dirname(__file__), 'ai_models', 'lead_scoring_pipeline.pkl')
    if os.path.exists(lead_model_path):
        LEAD_SCORING_MODEL = joblib.load(lead_model_path)
        logger.info("Successfully loaded Lead Scoring model.")
    else:
        logger.warning(f"Lead Scoring model not found at {lead_model_path}. Train the model first.")

    # 2. Load Churn Prediction Model (Placeholder setup)
    # churn_model_path = os.path.join(os.path.dirname(__file__), 'ai_models', 'churn_prediction_pipeline.pkl')
    # if os.path.exists(churn_model_path):
    #     CHURN_PREDICTION_MODEL = joblib.load(churn_model_path)
    #     logger.info("Successfully loaded Churn Prediction model.")

# -------------------------------------------------------------
# Data Schemas (Input Validation)
# -------------------------------------------------------------
class LeadFeatures(BaseModel):
    Lead_Origin: str
    Lead_Source: str
    Total_Time_Spent: float
    Occupation: str
    Student_Year: str  # "Year 1", "Year 2", "Year 3", "Year 4", or "Not Applicable"
    Specialization: str
    TotalVisits: float
    Page_Views: float

class StudentFeatures(BaseModel):
    Attendance_Rate: float
    Average_Score: float
    Number_of_Absences: int
    Course_Completion_Percent: float

# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.get("/")
def health_check():
    """
    Health check endpoint to verify the service routing.
    """
    return {
        "service": "SmartEdu_CRM_AI_Service",
        "status": "Healthy",
        "available_endpoints": ["/predict-lead", "/predict-churn"]
    }

@app.post("/predict-lead")
def predict_lead_scoring(features: LeadFeatures):
    """
    Generates a conversion probability score for a given lead.
    """
    if LEAD_SCORING_MODEL is None:
        raise HTTPException(status_code=500, detail="Lead Scoring model operates in offline mode. Missing PKL file.")
        
    try:
        # Convert incoming JSON payload to DataFrame for the sklearn pipeline
        input_data = pd.DataFrame([{
            'Lead Origin': features.Lead_Origin,
            'Lead Source': features.Lead_Source,
            'Total Time Spent on Website': features.Total_Time_Spent,
            'What is your current occupation': features.Occupation,
            'Specialization': features.Specialization,
            'TotalVisits': features.TotalVisits,
            'Page Views Per Visit': features.Page_Views,
            'Student_Year': features.Student_Year
        }])
        
        # Execute prediction
        prediction = LEAD_SCORING_MODEL.predict(input_data)[0]
        probability = LEAD_SCORING_MODEL.predict_proba(input_data)[0][1]
        
        return {
            "prediction": "Converted" if prediction == 1 else "Not Converted",
            "probability_score": round(probability * 100, 2),
            "recommendation": "HOT_LEAD" if probability >= 0.8 else ("WARM_LEAD" if probability >= 0.5 else "COLD_LEAD")
        }
    except Exception as e:
        logger.error(f"Error during lead scoring prediction: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to process prediction request. Check input feature types.")

@app.post("/predict-churn")
def predict_churn(features: StudentFeatures):
    """
    Placeholder endpoint for Churn Prediction routing demonstration.
    """
    # Simulated business logic based loosely on input features
    risk_level = "HIGH_RISK" if features.Number_of_Absences >= 3 else "LOW_RISK"
    
    return {
        "status": "Simulator Mode",
        "risk_level": risk_level,
        "message": "Endpoint mapped successfully. Awaiting model integration."
    }

# -------------------------------------------------------------
# Run Execution
# Terminal command: uvicorn api_server:app --reload
# -------------------------------------------------------------
