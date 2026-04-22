# Promoter Sequence Classifier

A 1D CNN trained on EPDnew human promoter sequences that classifies
600 bp DNA windows as **promoter** or **non-promoter** with a confidence score.

## Model Performance

| Metric         | Value  |
|----------------|--------|
| Test Accuracy  | 99.22% |
| Test Loss      | 0.0288 |
| Val Loss       | 0.0293 |
| Best Epoch     | 19/30  |
| Parameters     | 82,081 |

## Architecture

```
Input (4, 600)
→ Conv1d(4→32, k=11)  + BN + ReLU + MaxPool
→ Conv1d(32→64, k=7)  + BN + ReLU + MaxPool
→ Conv1d(64→128, k=7) + BN + ReLU + MaxPool
→ GlobalAvgPool
→ Dropout(0.5)
→ FC(128→64) + ReLU
→ FC(64→1) → Sigmoid
```

## Dataset

- Source: EPDnew human promoters (https://epd.expasy.org)
- Positives: ~29,000 real promoter sequences, 600 bp centered on TSS (-499 to +100)
- Negatives: Shuffled promoter sequences (preserves nucleotide frequency, destroys positional signal)
- Split: 80% train / 10% val / 10% test

## Sequence Length Handling

| Input Length  | Behavior                              | Reliability     |
|---------------|---------------------------------------|-----------------|
| == 600 bp     | Direct inference                      | Full            |
| > 600 bp      | Trimmed to center 600 bp              | Full            |
| 500–599 bp    | Padded symmetrically with N (zeros)   | Minor impact    |
| < 500 bp      | Rejected — error returned             | Unreliable      |

## Training Curves

![Training Curves](models/training_curves.png)

## Run Locally

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train model (skip if cnn_promoter.pt already exists)
cd src && python preprocess.py && python train.py

# Start API
cd api && uvicorn main:app --reload --port 8000
```

## Run with Docker

```bash
# Build
docker build -t promoter-classifier .

# Run
docker run -p 8000:8000 promoter-classifier
```

## API Endpoints

All endpoints use the `/api/v1/` prefix.

### POST /api/v1/classify
Classify a single DNA sequence.

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ATCG...600bp"}'
```

Response:
```json
{
  "label":             "promoter",
  "confidence":        0.9981,
  "probability":       0.9981,
  "original_length":   600,
  "processed_length":  600,
  "warning":           null,
  "time_ms":           4.21
}
```

### POST /api/v1/batch_classify
Classify multiple sequences (up to 100 per request).

```bash
curl -X POST http://localhost:8000/api/v1/batch_classify \
  -H "Content-Type: application/json" \
  -d '{"sequences": ["ATCG...600bp", "GCTA...600bp"]}'
```

### POST /api/v1/classify_file
Classify sequences from a FASTA file.

```bash
curl -X POST http://localhost:8000/api/v1/classify_file \
  -F "file=@sequences.fasta"
```

Supported file formats: `.fa`, `.fasta`, `.txt`  
Maximum 100 sequences per file.

### GET /api/v1/health
Health check and model status.

```bash
curl http://localhost:8000/api/v1/health
```

## Interactive Documentation

Visit the interactive API docs:
```
http://localhost:8000/docs
```

## Future Improvements
- Sliding window inference for whole genome sequences
- Attention visualization to highlight informative positions
- Multi-species promoter support
- ONNX export for faster CPU inference

## Dependencies

- torch==2.2.2
- numpy==1.26.4
- biopython==1.83
- scikit-learn==1.4.2
- matplotlib==3.8.4
- fastapi==0.111.0
- uvicorn==0.29.0
- pydantic==2.7.1
- requests==2.31.0
- jupyter==1.0.0