from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = "saved_models/distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="Phishing Email Detection API",
    description="Transformer-based phishing email detector (DistilBERT).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailRequest(BaseModel):
    subject: Optional[str] = None
    body: str


class PredictionResponse(BaseModel):
    prediction_label: str
    prediction_int: int
    prob_ham: float
    prob_phishing: float
    model_name: str

tokenizer: AutoTokenizer | None = None
model: AutoModelForSequenceClassification | None = None

@app.on_event("startup")
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print("✅ Model & tokenizer loaded on startup.")

@app.get("/health")
def health_check():
    return {"status": "ok"}

def predict_email(subject: Optional[str], body: str) -> PredictionResponse:
    text = f"{(subject or '').strip()} {body.strip()}".strip()

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred_int = int(np.argmax(probs))
    label = "phishing" if pred_int == 1 else "ham"

    return PredictionResponse(
        prediction_label=label,
        prediction_int=pred_int,
        prob_ham=float(probs[0]),
        prob_phishing=float(probs[1]),
        model_name=MODEL_PATH.split("/")[-1],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: EmailRequest):
    return predict_email(request.subject, request.body)
