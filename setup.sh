#!/bin/bash

echo "Setting up Titanic ML Application..."

mkdir -p datasets models plots logs data/raw data/processed

if [ ! -f "datasets/TitanicDataset.csv" ]; then
    echo "Warning: TitanicDataset.csv not found in datasets/"
    echo "Please place your dataset in datasets/TitanicDataset.csv"
    echo "Or use --data-path argument to specify custom path"
fi

echo "Starting application..."
streamlit run main.py -- "$@"
