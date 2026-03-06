from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Input(BaseModel):
    number: int

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(x: Input):
    return {"is_even": x.number % 2 == 0}

