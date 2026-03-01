from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. Configuration: Model file settings
# Ensure 'loan_model.joblib' exists in the same directory
MODEL_FILE = 'loan_model.joblib'

app = FastAPI()

# 2. CORS configuration for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Model Ingestion: Load the pre-trained RandomForest model
try:
    model = joblib.load(MODEL_FILE)
    # Extract feature names required by the model during training
    model_columns = model.feature_names_in_
    print(f"SUCCESS: Loaded {MODEL_FILE}")
except Exception as e:
    print(f"ERROR: Failed to load model file ({MODEL_FILE}): {e}")

# 4. Request Schema: Defines the input features for loan evaluation
class LoanPredictionInput(BaseModel):
    limit_balance: float
    age: int
    bill_amt_1: float
    pay_amt_1: float
    sex: str         # Expected values: '1' (Male), '2' (Female)
    education: str   # Expected values: '1', '2', '3', etc.
    marriage: str    # Expected values: '1', '2', '3'

@app.get("/")
def read_root():
    # Health check endpoint
    return {"status": "Loan Approval API is active", "model": MODEL_FILE}

@app.post("/predict")
def predict(data: LoanPredictionInput):
    # Convert incoming JSON data to a Pandas DataFrame
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    
    # 5. Preprocessing: Categorical Encoding
    # Transform categorical strings into dummy variables (One-Hot Encoding)
    input_df = pd.get_dummies(input_df, columns=['sex', 'education', 'marriage'])
    
    # 6. Column Alignment: Matching features with the training phase
    # Create missing columns with 0s and ensure identical column order
    final_df = pd.DataFrame(columns=model_columns)
    final_df = pd.concat([final_df, input_df], sort=False).fillna(0)
    final_df = final_df[model_columns]
    
    # 7. Execution: Model Inference
    # prediction: 0 (Approved) or 1 (Default/Rejected)
    # probability: Confidence score of the default class
    prediction = int(model.predict(final_df)[0])
    probability = float(model.predict_proba(final_df)[0][1])
    
    # Apply a strict threshold of 0.1 for banking risk management
    # If the probability is 10% or higher, the loan is rejected (is_default = 1)
    prediction = 1 if probability >= 0.15 else 0

    # Return JSON response with structured results
    return {
        "is_default": prediction,
        "default_probability": round(probability, 4),
        "status": "Rejected" if prediction == 1 else "Approved"
    }

# Entry point for Cloud Run (Default port is 8080)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)