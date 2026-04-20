#!/bin/bash

echo "Setting up Promoter CNN project..."

# Create directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p src
mkdir -p api
mkdir -p models
mkdir -p notebooks

# Create source files
touch src/__init__.py
touch src/preprocess.py
touch src/dataset.py
touch src/model.py
touch src/train.py
touch src/utils.py

# Create API files
touch api/__init__.py
touch api/main.py
touch api/inference.py

# Create root files
touch requirements.txt
touch .gitignore
touch Dockerfile
touch README.md

echo "Done. Directory structure created."