import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.routing import APIRouter
from pydantic import BaseModel, field_validator
from typing import Optional
import time
from io import StringIO
from Bio import SeqIO
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
        "1D CNN classifier trained on EPDnew human promoter sequences. "
        "Accepts DNA sequences between 500–600+ bp and classifies as "
        "promoter or non-promoter with confidence score."
    ),
    version="1.0.0",
    lifespan=lifespan
)

router = APIRouter(prefix="/api/v1")

class SequenceRequest(BaseModel):
    sequence: str

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        v = v.replace('\n', '').replace('\r', '').replace(' ', '').upper().strip()
        
        if len(v) == 0:
            raise ValueError("Sequence cannot be empty.")
        
        if " " in v:
            raise ValueError("Sequence cannot contain spaces.")
        
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
    original_length: int
    processed_length: int
    warning: Optional[str]
    time_ms: float


class BatchResponse(BaseModel):
    total: int
    results: list[dict]
    time_ms: float


@app.get("/")
def root():
    return {"message": "Promoter Sequence Classifier API is running. Visit /docs for the API documentation."}


@router.get("/health")
def health_check():
     return {
        "status": "ok",
        "model_loaded": "model" in model_store,
        "accepted_length_range": f"{500}–{SEQUENCE_LENGTH}+ bp",
        "trained_on": f"{SEQUENCE_LENGTH} bp EPDnew human promoters"
    }


@router.post("/classify", response_model=PredictionResponse)
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
    

@router.post("/batch_classify", response_model=BatchResponse)
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


@router.post("/classify_file")
async def classify_file(file: UploadFile = File(...)):
    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    allowed_extensions = {".fa", ".fasta", ".txt"}
    filename = file.filename or "upload"
    extension = os.path.splitext(filename)[-1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {extension}. "
                   f"Accepted: {allowed_extensions}"
        )

    try:
        content = await file.read()
        text = content.decode("utf-8")
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read file: {str(e)}"
        )

    try:
        fasta_io  = StringIO(text)
        records = list(SeqIO.parse(fasta_io, "fasta"))
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse FASTA file: {str(e)}"
        )

    if len(records) == 0:
        raise HTTPException(
            status_code=422,
            detail="No sequences found in file. Make sure it is valid FASTA format."
        )

    if len(records) > 100:
        raise HTTPException(
            status_code=422,
            detail=f"Too many sequences: {len(records)}. Maximum 100 per upload."
        )

    try:
        start = time.perf_counter()
        results = []

        for i, record in enumerate(records):
            seq = str(record.seq).upper().strip()
            try:
                result = predict_sequence(seq, model_store["model"])
                result["index"] = i
                result["record_id"] = str(record.id)
                results.append(result)

            except ValueError as e:
                results.append({
                    "index": i,
                    "record_id": str(record.id),
                    "label": "error",
                    "message": str(e)
                })

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        return {
            "filename": filename,
            "total": len(results),
            "results": results,
            "time_ms": elapsed_ms
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )
    

app.include_router(router)