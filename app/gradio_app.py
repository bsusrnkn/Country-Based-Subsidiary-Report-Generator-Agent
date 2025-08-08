# Author: Bengisu Serinken
# August 2025
from geopy.geocoders import Nominatim
from time import sleep
from weasyprint import HTML
from eventregistry import *
from pydantic import BaseModel
import gradio as gr
import app.AgentDefinition as AgentDefinition
import plotly.graph_objects as go
import pycountry, datetime, descriptions, os

def generate_html_from_data(input_data: dict) -> str:
    # Format of the input data:
    # pdf_input = {
    #     "country_name": data_blob.country_name,
    #     "gdp_data": data_blob.gdp_data or {},
    #     "exchange_data": data_blob.exchange_data or {},
    #     "continent": data_blob.continent or "",
    #     "summary": summary_output.summary,
    #     "unemployment_data": data_blob.unemployment_data or {}
    # }
    print("INSIDE GENERATE_PDF_FROM_HTML")
    
    if not isinstance(input_data, dict):
        return f"Invalid input format: expected dict"

    try:
        country_name = input_data["country_name"]
        raw_gdp_data = input_data["gdp_data"]
        raw_exchange = input_data["exchange_data"]
        continent    = input_data["continent"]
        summary      = input_data["summary"]
        raw_unemployment_data = input_data["unemployment_data"]
    except KeyError as e:
        return f"Missing required key in input: {e}"
    
    if isinstance(raw_gdp_data, BaseModel):
        gdp_data = raw_gdp_data.model_dump()
    else:
        gdp_data = raw_gdp_data or {}

    if isinstance(raw_exchange, BaseModel):
        exchange_data = raw_exchange.model_dump()
    else:
        exchange_data = raw_exchange or {}
    
    if isinstance(raw_unemployment_data, BaseModel):
        unemployment_data = raw_unemployment_data.model_dump()
    else:
        unemployment_data = raw_unemployment_data or {}

    gdp_items = []
    for key, value in gdp_data.items():
        desc = descriptions.gdp_descriptions.get(key, "") 
        gdp_items.append(f"<li><strong>{desc if desc else key.replace('_', ' ').title()}</strong>: {value}</li>")
    
    exchange_items = []
    for key, value in exchange_data.items():
        desc = descriptions.fx_descriptions.get(key, "") 
        exchange_items.append(f"<li><strong>{desc if desc else key.replace('_', ' ').title()}</strong>: {value}</li>")  

    unemployment_items = []
    for key, value in unemployment_data.items():
        desc = descriptions.unemployment_descriptions.get(key, "") 
        unemployment_items.append(f"<li><strong>{desc if desc else key.replace('_', ' ').title()}</strong>: {value}</li>")
    
    start_keyword = "</think>"
    if start_keyword in summary:
        start_index = summary.index(start_keyword) + len(start_keyword)
        summary = summary[start_index:] 

    # HTML for PDF Report
    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ color: #000000;font-size: 14px; font-family: Tahoma; text-align: justify; line-height: 1.6;}}
            h1 {{ color: #000000; font-size: 23px; font-family: Tahoma; text-align: justify; line-height: 1.6;}}
            h2 {{ color: #000000; font-size: 20px; font-family: Tahoma; text-align: justify; line-height: 1.6;}}
            h3 {{ color: #000000; font-size: 17px; font-family: Tahoma; text-align: justify; line-height: 1.6;}}
            p {{ color: #000000;font-size: 14px; font-family: Tahoma; text-align: justify; line-height: 1.6;}}
            ul {{ color: #000000; font-size: 14px; font-family: Tahoma; text-align: justify; line-height: 1.6;}}
        </style>
    </head>
    <body>
        <p>Date: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}</p>
        <h1>Country Report: {country_name}</h1>

        {f"<ul><strong>Continent:</strong> {continent}</ul>" if continent else ""}

        {f"<h2>GDP Information</h2><ul>{''.join(gdp_items)}</ul>" if gdp_items else ""}
        {f"<h2>Exchange Rate Information</h2><ul>{''.join(exchange_items)}</ul>" if exchange_items else ""}
        {f"<h2>Unemployment Rate Information</h2><ul>{''.join(unemployment_items)}</ul>" if unemployment_items else ""}
        <h2>Summary</h2>
        {summary if summary else "<p>No summary generated.</p>"}
    
    </body>
    </html>
    """
    print(f"END OF GENERATE_PDF_FROM_HTML")
    return html

geolocator = Nominatim(
    user_agent="subsidiary-report-generator",
    timeout=10 
)

def generate_report(country, langugage):
    try:
        print(f"Generating report for {country}...")
        input_data = AgentDefinition.run_pipeline(country)
        print("1")
        html = generate_html_from_data(input_data)
        print("HTML generated successfully.")
        
        if(langugage == "Turkish"):
            html = AgentDefinition.translate_html_to_turkish(html)
            html = html.strip().splitlines()
            if html and html[0].startswith("```"):
                html = html[1:]
            if html and html[-1].strip().startswith("```"):
                html = html[:-1]
            html = "\n".join(html)
            
        print("Language translation completed.")
        
        filename = f"{country}_Subsidiary_Report.pdf"
        HTML(string=html).write_pdf(filename)
        pdf_path = filename

        print("2")
        ui_summary = f"Successfully generated report for {country}\nPDF saved to: {pdf_path}"
        print("3")
        return ui_summary, pdf_path
    except Exception as e:
        error_msg = f"Error generating report: {str(e)}"
        return error_msg, None
    
# Function to get coordinates dynamically
def get_country_coordinates(country_name):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(country_name)
            if location:
                return location.latitude, location.longitude
            sleep(1)  # Add delay between retries
        except Exception as e:
            print(f"Geocoding attempt {attempt + 1} failed: {str(e)}")
            sleep(1)
    return None, None
  
# Plotly map generator using dynamic coordinates
def generate_map(country):
    lat, lon = get_country_coordinates(country)
    if lat is None or lon is None:
        lat, lon = 20.0, 0.0 
    fig = go.Figure(go.Scattermap(
        lat=[lat],
        lon=[lon],
        mode='markers',
        marker=go.scattermap.Marker(size=12),
        text=[country],
        hoverinfo="text"
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=lat, lon=lon), zoom=3),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

countries = sorted([country.name for country in pycountry.countries])

with gr.Blocks() as demo:
    gr.Markdown("## Country Based Subsidiary Report Generator")
    with gr.Row():
        country_dropdown = gr.Dropdown(choices=countries, label="Type or Select a Country", value=None)
        language_dropdown = gr.Dropdown(choices=["English", "Turkish"], label="Select Report Language", value="English")
        generate_btn = gr.Button("Generate Report")
    
    with gr.Row():
        report_output = gr.Textbox(label="Report Status")
        map_plot = gr.Plot(label="Country on World Map")
        pdf_download = gr.File(label="Download PDF Report")


        def update_outputs(selected_country, selected_language):
            ui_summary, pdf_path = generate_report(selected_country, selected_language)
            map_fig = generate_map(selected_country)
            
            if pdf_path and os.path.exists(pdf_path):
                return ui_summary, map_fig, pdf_path
            return ui_summary, map_fig, None
            
        generate_btn.click(
            fn=update_outputs,
            inputs=[country_dropdown, language_dropdown],
            outputs=[report_output, map_plot, pdf_download]
        )
        demo.launch()
