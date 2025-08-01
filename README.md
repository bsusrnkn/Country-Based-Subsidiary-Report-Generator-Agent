# Country-Based Subsidiary Report Generator Agent

This project generates automated investment reports for selected countries using local LLMs (Qwen3 via Ollama and gemini-2.0-flash-lite via GoogleGenerativeAI) for creating an agent using LangChain. Gradio is used as UI to take user inputs and display outputs. HTML is used for report generation.

## Features

- Fetches real-time macroeconomic & investment metrics 
- Uses Qwen3 LLM to generate an analysis
- Displays country location on interactive map
- Prepares the report as HTML and displays as a downloadable PDF

## How to Run

1. Activate your virtual environment: 

source venv/bin/activate


2. Install requirements:

pip install -r requirements.txt


2. Launch the app:

python3 app/gradio_app.py

