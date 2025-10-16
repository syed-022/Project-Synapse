# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import json

from ml_handler import MLHandler

app = FastAPI()

# --- Middleware ---
# Allow requests from your frontend (important for development and deployment)
origins = [
    "http://localhost:3000", # Local React dev server
    "https://synapse-frontend.onrender.com", # Your deployed frontend URL
    # Add your custom domain here if you have one
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For simplicity, allow all. For production, restrict to `origins` list.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory "database" ---
# For this project, we'll keep the trained model in memory.
ml_handler = None
dataset_columns = []

class PredictionRequest(BaseModel):
    data: dict

class ChatRequest(BaseModel):
    question: str
    analysis_context: dict

@app.get("/")
def read_root():
    return {"message": "Synapse AI Backend is running."}

@app.post("/api/analyze")
async def analyze_data(file: UploadFile = File(...)):
    global ml_handler, dataset_columns
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        ml_handler = MLHandler(df)
        results = ml_handler.run_analysis()
        dataset_columns = ml_handler.get_feature_names()

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    if not ml_handler:
        raise HTTPException(status_code=400, detail="No model trained. Please analyze a dataset first.")
    
    try:
        input_df = pd.DataFrame([request.data])
        prediction, importance = ml_handler.predict_and_explain(input_df)

        return {
            "prediction": prediction,
            "feature_importance": importance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/chat")
async def chat_with_assistant(request: ChatRequest):
    """
    This endpoint simulates a call to a generative AI model (like Gemini)
    to answer user questions about the analysis.
    """
    question = request.question.lower()
    context = request.analysis_context
    
    # --- SIMULATED GEMINI API CALL LOGIC ---
    if "best model" in question:
        best_model = context.get('bestModel', 'the one with the highest accuracy')
        response = f"The best performing model was the **{best_model}**. It was selected because it achieved the highest accuracy and a strong F1-score, indicating a good balance between precision and recall."
    elif "explain" in question and "xgboost" in question:
        response = "Certainly. **XGBoost (eXtreme Gradient Boosting)** is a powerful algorithm that builds decision trees sequentially, where each new tree corrects the errors of the previous one. It's highly effective due to its performance and built-in regularization, which helps prevent overfitting."
    elif "suggestion" in question or "improve" in question:
        response = "To potentially improve model performance, you could consider: <br>1. **Feature Engineering:** Creating new features from existing ones. <br>2. **Hyperparameter Tuning:** Searching for the best model parameters. <br>3. **Gathering More Data:** More high-quality data often leads to better models."
    elif "what is shap" in question or "drivers" in question:
        response = "The 'Decision Drivers' are calculated using **SHAP (SHapley Additive exPlanations)**. It's a method from game theory that explains how much each feature (like 'Glucose') contributed to pushing a specific prediction away from the average. A larger bar means that feature had a bigger impact."
    else:
        response = "I can help with that. Could you please rephrase your question? You can ask me to explain a model, suggest improvements, or define a term like 'F1-score'."

    return {"answer": response}
