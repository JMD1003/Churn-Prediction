from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.inference import predict

app = FastAPI()

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
def api_predict(data: CustomerData):
    try:
        out = predict(data.dict())
        return {"prediction": out}
    except Exception as e:
        return {"error": str(e)}

def gradio_interface(
    Gender, Age, Geography, CreditScore, Tenure,
    Balance, NumOfProducts, EstimatedSalary,
    HasCrCard, IsActiveMember, Complain,
    Satisfaction_Score, Point_Earned, Card_Type
):
    payload = {
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
    out = predict(payload)
    return str(out)

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Number(label="Age"),
        gr.Dropdown(["France", "Germany", "Spain"], label="Geography"),
        gr.Number(label="Credit Score"),
        gr.Number(label="Tenure (years)"),
        gr.Number(label="Balance ($)"),
        gr.Number(label="Number of Products"),
        gr.Number(label="Estimated Salary ($)"),
        gr.Dropdown([0, 1], label="Has Credit Card (0=No, 1=Yes)"),
        gr.Dropdown([0, 1], label="Is Active Member (0=No, 1=Yes)"),
        gr.Dropdown([0, 1], label="Complain (0=No, 1=Yes)"),
        gr.Dropdown([1, 2, 3, 4, 5], label="Satisfaction Score"),
        gr.Number(label="Points Earned"),
        gr.Dropdown(["SILVER", "GOLD", "PLATINUM", "DIAMOND"], label="Card Type"),
    ],
    outputs="text",
    title="Bank Churn Predictor",
    description="Fill in the customer details to get a churn prediction.",
)


app = gr.mount_gradio_app(app, demo, path="/ui")