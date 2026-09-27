# Loan Approval Prediction

Reproducible classification project demonstrating a loan-approval workflow without using private or real applicant records.

## Dataset
The training script generates a clearly synthetic dataset with a fixed random seed. This keeps the repository self-contained and avoids presenting fabricated records as real banking data.

## Models
Logistic Regression · Random Forest · Gradient Boosting

## Metrics
Accuracy · Precision · Recall · F1

## Run
```bash
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python src/train.py
```

## Methodology
The project demonstrates preprocessing with numeric scaling, categorical one-hot encoding, stratified splitting, model comparison, and Joblib export.

## Important limitation
This is an educational ML workflow. It must not be used as a real credit or lending decision system.

## Author
Hassan Ali — Computer Science student focused on Machine Learning and AI Engineering.
