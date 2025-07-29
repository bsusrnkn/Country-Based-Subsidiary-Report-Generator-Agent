import gradio as gr
import pycountry
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import AgentDefinition as AgentDefinition
import requests, os
from time import sleep

# Initialize geolocator with longer timeout and retries
geolocator = Nominatim(
    user_agent="subsidiary-report-generator",
    timeout=10  # Increased timeout
)

def generate_report(country):
    try:
        pdf_path = AgentDefinition.agent_invoking_function(country)
        summary = f"Successfully generated report for {country}\nPDF saved to: {pdf_path}"
        return summary, pdf_path
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
#currencies = [fetch_currency_code(c) for c in countries]
#currencies = ['EUR', 'USD', 'XCD', 'XOF', 'AUD', 'XAF', 'GBP', 'NZD', 'DKK', 'XPF', 'DZD', 'ANG', 'EGP', 'CHF', 'NOK', 'AFN', 'ALL', 'AOA', 'ARS', 'AMD', 'AWG', 'AZN', 'BSD', 'BHD', 'BDT', 'BBD', 'BYN', 'BZD', 'BMD', 'BTN', 'BOB', 'BAM', 'BWP', 'BRL', 'BND', 'BGN', 'BIF', 'CVE', 'KHR', 'CAD', 'KYD', 'CLP', 'CNY', 'COP', 'KMF', 'CDF', 'CKD', 'CRC', 'CUC', 'CZK', 'DJF', 'DOP', 'ERN', 'SZL', 'ETB', 'FKP', 'FJD', 'GMD', 'GEL', 'GHS', 'GIP', 'GTQ', 'GNF', 'GYD', 'HTG', 'HNL', 'HKD', 'HUF', 'ISK', 'INR', 'IDR', 'IRR', 'IQD', 'ILS', 'JMD', 'JPY', 'JOD', 'KZT', 'KES', 'KPW', 'KRW', 'KWD', 'KGS', 'LAK', 'LBP', 'LSL', 'LRD', 'LYD', 'MOP', 'MGA', 'MWK', 'MYR', 'MVR', 'MRU', 'MUR', 'MXN', 'MDL', 'MNT', 'MAD', 'MZN', 'MMK', 'NAD', 'NPR', 'NIO', 'NGN', 'MKD', 'OMR', 'PKR', 'PAB', 'PGK', 'PYG', 'PEN', 'PHP', 'PLN', 'QAR', 'RON', 'RUB', 'RWF', 'WST', 'STN', 'SAR', 'RSD', 'SCR', 'SLE', 'SGD', 'SBD', 'SOS', 'ZAR', 'SSP', 'LKR', 'SDG', 'SRD', 'SEK', 'SYP', 'TWD', 'TJS', 'TZS', 'THB', 'TOP', 'TTD', 'TND', 'TMT', 'TRY', 'UGX', 'UAH', 'AED', 'UYU', 'UZS', 'VUV', 'VES', 'VND', 'YER', 'ZMW', 'ZWL']

with gr.Blocks() as demo:
    gr.Markdown("## Country Based Subsidiary Report Generator")
    with gr.Row():
        country_dropdown = gr.Dropdown(choices=countries, label="Type or Select a Country", value=None)
        generate_btn = gr.Button("Generate Report")
    
    with gr.Row():
        report_output = gr.Textbox(label="Report Status")
        map_plot = gr.Plot(label="Country on World Map")
        pdf_download = gr.File(label="Download PDF Report")

        def update_outputs(selected_country):
            summary, pdf_path = generate_report(selected_country)
            map_fig = generate_map(selected_country)
            
            if pdf_path and os.path.exists(pdf_path):
                return summary, map_fig, pdf_path
            return summary, map_fig, None
            
        generate_btn.click(
            fn=update_outputs,
            inputs=[country_dropdown],
            outputs=[report_output, map_plot, pdf_download]
        )
        demo.launch()
