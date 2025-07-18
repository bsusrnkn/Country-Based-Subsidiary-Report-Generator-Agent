import gradio as gr
import pycountry
import plotly.graph_objects as go
from geopy.geocoders import Nominatim

# Initialize geolocator with a user agent
geolocator = Nominatim(user_agent="subsidiary-report-generator")

# Function to get coordinates dynamically
def get_country_coordinates(country_name):
    location = geolocator.geocode(country_name)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

# Get full list of country names using pycountry
countries = sorted([country.name for country in pycountry.countries])

# Dummy report generation function
def generate_report(country):
    return f"Sample report for {country}. More data coming soon."

# Plotly map generator using dynamic coordinates
def generate_map(country):
    lat, lon = get_country_coordinates(country)
    if lat is None or lon is None:
        lat, lon = 20.0, 0.0  # fallback location
    fig = go.Figure(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode='markers',
        marker=go.scattermapbox.Marker(size=12),
        text=[country],
        hoverinfo="text"
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=lat, lon=lon), zoom=3),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Country-Based Subsidiary Report Generator")
    with gr.Row():
        country_dropdown = gr.Dropdown(choices=countries, label="Type or Select a Country")
        generate_btn = gr.Button("Generate Report")
    report_output = gr.Textbox(label="Generated Report")
    map_plot = gr.Plot(label="Country on World Map")

    def update_outputs(selected_country):
        return generate_report(selected_country), generate_map(selected_country)

    generate_btn.click(fn=update_outputs, inputs=[country_dropdown], outputs=[report_output, map_plot])

demo.launch()
