import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict

app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="ML API for predicting customer churn in banking",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok"}

class CustomerData(BaseModel):

    Gender: str           
    Age: int              
    Geography: str         

    CreditScore: int       
    Tenure: int            
    Balance: float        
    NumOfProducts: int     
    EstimatedSalary: float 

    HasCrCard: int        
    IsActiveMember: int    
    Complain: int          
    Satisfaction_Score: int  
    Point_Earned: int    
    Card_Type: str         

@app.post("/predict")
def get_prediction(data: CustomerData):
    try:
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}


def gradio_interface(
    Gender, Age, Geography, CreditScore, Tenure,
    Balance, NumOfProducts, EstimatedSalary,
    HasCrCard, IsActiveMember, Complain,
    Satisfaction_Score, Point_Earned, Card_Type
):
    data = {
        "Gender": Gender,
        "Age": int(Age),
        "Geography": Geography,
        "CreditScore": int(CreditScore),
        "Tenure": int(Tenure),
        "Balance": float(Balance),
        "NumOfProducts": int(NumOfProducts),
        "EstimatedSalary": float(EstimatedSalary),
        "HasCrCard": int(HasCrCard),
        "IsActiveMember": int(IsActiveMember),
        "Complain": int(Complain),
        "Satisfaction_Score": int(Satisfaction_Score),
        "Point_Earned": int(Point_Earned),
        "Card_Type": Card_Type,
    }
    result = predict(data)
    return str(result)

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Number(label="Age", value=35, minimum=18, maximum=100),
        gr.Dropdown(["France", "Germany", "Spain"], label="Geography", value="France"),
        gr.Number(label="Credit Score", value=650, minimum=300, maximum=850),
        gr.Number(label="Tenure (years)", value=5, minimum=0, maximum=10),
        gr.Number(label="Balance ($)", value=75000.0, minimum=0),
        gr.Number(label="Number of Products", value=1, minimum=1, maximum=4),
        gr.Number(label="Estimated Salary ($)", value=50000.0, minimum=0),
        gr.Dropdown([0, 1], label="Has Credit Card (0=No, 1=Yes)", value=1),
        gr.Dropdown([0, 1], label="Is Active Member (0=No, 1=Yes)", value=1),
        gr.Dropdown([0, 1], label="Complain (0=No, 1=Yes)", value=0),
        gr.Dropdown([1, 2, 3, 4, 5], label="Satisfaction Score", value=3),
        gr.Number(label="Points Earned", value=500, minimum=0),
        gr.Dropdown(["SILVER", "GOLD", "PLATINUM", "DIAMOND"], label="Card Type", value="SILVER"),
    ],
    outputs=gr.Textbox(label="Churn Prediction", lines=2),
    title="Bank Customer Churn Predictor",
    description="""
    **Predict customer churn probability using machine learning**
    
    Fill in the customer details below to get a churn prediction. The model uses XGBoost 
    trained on historical bank customer data to identify customers at risk of leaving.
    
    **Tip**: Older customers with low satisfaction scores and complaints tend to have 
    higher churn rates.
    """,
    examples=[

        ["Female", 45, "Germany", 600, 2, 120000.0, 1, 60000.0, 1, 0, 1, 1, 300, "SILVER"],

        ["Male", 32, "France", 750, 8, 50000.0, 2, 80000.0, 1, 1, 0, 5, 800, "GOLD"],
    ],    
)

app = gr.mount_gradio_app(app, demo, path="/ui")