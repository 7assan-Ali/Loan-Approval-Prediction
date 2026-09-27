from pathlib import Path
import pandas as pd
from joblib import load
ROOT=Path(__file__).resolve().parents[1]; model=load(ROOT/"models/model.joblib")
sample=pd.DataFrame([{"income_score":0.4,"credit_history":0.8,"loan_amount_score":-0.2,"employment_score":0.7,"age_score":0.1,"debt_score":-0.3,"employment_type":"employed"}])
print("Prediction:",model.predict(sample)[0])
