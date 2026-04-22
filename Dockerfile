FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the API needs at runtime
COPY src/model.py        ./src/model.py
COPY src/utils.py        ./src/utils.py
COPY api/inference.py    ./api/inference.py
COPY api/main.py         ./api/main.py
COPY models/cnn_promoter.pt ./models/cnn_promoter.pt

# Set PYTHONPATH so api/ can import from src/
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]