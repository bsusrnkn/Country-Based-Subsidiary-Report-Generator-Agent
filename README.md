# Country-Based Subsidiary Report Generator Agent

This project generates automated investment reports for selected countries using a local LLM (Qwen3 via Ollama), and presents it via a Gradio UI.

## Features

- Fetches real-time macroeconomic & investment metrics 
- Uses Qwen3 LLM to generate LaTeX-based analysis
- Displays country location on interactive map
- Prepares professional LaTex reports into PDF

## How to Run

1. Activate your virtual environment: 

source venv/bin/activate


2. Install requirements:

pip install -r requirements.txt


2. Launch the app:

python app/gradio_app.py

