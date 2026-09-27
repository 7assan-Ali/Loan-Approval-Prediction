from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
ROOT=Path(__file__).resolve().parents[1]; MODEL_DIR=ROOT/"models"; MODEL_DIR.mkdir(exist_ok=True)
Xn,y=make_classification(n_samples=1500,n_features=6,n_informative=4,n_redundant=1,weights=[.35,.65],random_state=42)
X=pd.DataFrame(Xn,columns=["income_score","credit_history","loan_amount_score","employment_score","age_score","debt_score"])
X["employment_type"]=np.where(X["employment_score"]>0,"employed","other"); y=pd.Series(y,name="loan_approved")
num=X.select_dtypes(include=np.number).columns.tolist(); cat=X.select_dtypes(exclude=np.number).columns.tolist()
pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),num),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
models={"Logistic Regression":LogisticRegression(max_iter=2000),"Random Forest":RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1),"Gradient Boosting":GradientBoostingClassifier(random_state=42)}
for name,est in models.items():
    pipe=Pipeline([("preprocess",pre),("model",est)]); pipe.fit(Xt,yt); pred=pipe.predict(Xv)
    print(name,{"accuracy":accuracy_score(yv,pred),"precision":precision_score(yv,pred,zero_division=0),"recall":recall_score(yv,pred,zero_division=0),"f1":f1_score(yv,pred,zero_division=0)})
final=Pipeline([("preprocess",pre),("model",models["Random Forest"])]); final.fit(Xt,yt); dump(final,MODEL_DIR/"model.joblib")
