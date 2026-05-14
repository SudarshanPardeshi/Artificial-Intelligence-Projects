from fastapi import FastAPI
from pydantic import BaseModel

from inference import predict_melting_point


app = FastAPI(
    title="Melting Point Prediction API"
)


class MoleculeInput(BaseModel):
    smiles: str


@app.get("/")
def home():
    return {
        "message": "Melting Point Prediction API Running"
    }


@app.post("/predict")
def predict(data: MoleculeInput):

    prediction = predict_melting_point(
        data.smiles
    )

    return {
        "smiles": data.smiles,
        "predicted_melting_point": round(prediction, 2)
    }