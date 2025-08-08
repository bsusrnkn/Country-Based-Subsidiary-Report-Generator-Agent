# Country-Based Subsidiary Report Generator Agent

This project generates automated investment reports for selected countries by collecting macroeconomic and investment-related metrics via APIs and summarizing them using an AI agent built with LangChain and Gemini LLM. Gradio is used for the user interface, and the final report is generated in HTML and exported as a PDF.

## Features

* Fetches real-time macroeconomic and investment metrics from multiple APIs:

  * **API Ninjas** – GDP and unemployment rate
  * **ExchangeRate** – Currency exchange rates
  * **EventRegistry** – Latest news articles
  * **RestCountries** – Currency codes
  * **Nominatim** – Geographic coordinates
  * **pycountry & pycountry-convert** – Country and continent data
* Uses **Gemini 2.0 Flash Lite** (via GoogleGenerativeAI) for agent-driven data processing and summarization
* Dynamically selects tools based on a country’s development level
* Handles JSON parsing issues and API inconsistencies with **Pydantic BaseModel validation** and **OutputFixingParser**
* Supports Turkish translation of reports if selected by the user
* Displays country location on an interactive map
* Generates an HTML-based report and provides it as a downloadable PDF

## How It Works

1. **Tool Selection** – Based on the country’s development level, the most relevant data-fetching tools are chosen.
2. **Data Collection** – APIs are called to gather GDP, unemployment, exchange rates, news, coordinates, and other metrics.
3. **Summarization** – Gemini LLM processes and summarizes the collected data.
4. **Report Generation** – HTML is created from summarized content and existing data, optionally translated to Turkish by LLM, and exported to PDF.

## How to Run

1. Activate your virtual environment:

   ```bash
   source venv/bin/activate
   ```

2. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the app:

   ```bash
   python3 app/gradio_app.py
   ```
