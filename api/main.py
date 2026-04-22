from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional
import time
from inference import load_model, predict_sequence, batch_predict
from utils import SEQUENCE_LENGTH


model_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up - model loading...")
    model_store["model"] = load_model()
    print("Model ready.")
    yield
    
    model_store.clear()
    print("Shutting down...")


app = FastAPI(
    title="Promoter Sequence Classifier API",
    description=(
        "1D CNN trained on EPDnew human promoter sequences. "
        "Classifies 600 bp DNA windows as promoter or non-promoter."
    ),
    version="1.0.0",
    lifespan=lifespan
)


class SequenceRequest(BaseModel):
    sequence: str

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        v = v.upper().strip()
        valid_bases = set("ACGT")
        
        invalid_bases = set(v) - valid_bases
        if invalid_bases:
            raise ValueError(f"Sequence contains invalid bases: {invalid_bases}")

        return v
    

class BatchRequest(BaseModel):
    sequences: list[str]

    @field_validator("sequences")
    @classmethod
    def validate_batch_size(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("Batch must contain at least one sequence.")
        if len(v) > 100:
            raise ValueError("Maximum 100 sequences per batch request.")
        
        return v
    

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probability: float
    length: int
    time_ms: Optional[float] = None


class BatchResponse(BaseModel):
    total: int
    results: list[dict]
    time_ms: float


@app.get("/")
def root():
    return {"message": "Promoter Sequence Classifier API is running. Visit /docs for the API documentation."}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model_store.get("model") is not None,
        "sequence_length_expected": SEQUENCE_LENGTH,
    }


@app.post("/classify", response_model=PredictionResponse)
def classify(request: SequenceRequest):
    if "model" not in model_store:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )
    
    try:
        start_time = time.perf_counter()
        result = predict_sequence(request.sequence, model_store["model"])
        elapsed_time = round((time.perf_counter() - start_time) * 1000, 2)
        result["time_ms"] = elapsed_time

        return result
    
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )
    

@app.post("/batch_classify", response_model=BatchResponse)
def batch_classify(request: BatchRequest):
    if "model" not in model_store:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )
    
    try:
        start_time = time.perf_counter()
        result = batch_predict(request.sequences, model_store["model"])
        elapsed_time = round((time.perf_counter() - start_time) * 1000, 2)
        
        return {
            "total": len(request.sequences),
            "results": result,
            "time_ms": elapsed_time
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch inference error: {str(e)}"
        )