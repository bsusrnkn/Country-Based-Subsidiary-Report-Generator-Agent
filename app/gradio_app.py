import gradio as gr
import pycountry
import pycountry_convert as pc
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from agent import CountryReportAgent

agent = CountryReportAgent()


def generate_report(country):
    results = agent.tool  
    gdp_data = results.fetch_gdp(country)
    fx_data = results.fetch_exchange_rates(country)
    news_data = results.fetch_economic_news(country, max_articles=5)
    pdf_path = results.generate_pdf_from_html(country, gdp_data, fx_data, news_data)
    
    # For UI display in textbox
    summary = f"GDP Nominal: {gdp_data.get('gdp_nominal')} Billion\nExchange Rate to USD: {fx_data.get('conversion_rate_USD')}"
    
    return summary, pdf_path


# Initialize geolocator with a user agent
geolocator = Nominatim(user_agent="subsidiary-report-generator")

# Function to get coordinates dynamically
def get_country_coordinates(country_name):
    location = geolocator.geocode(country_name)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None
    
def get_continent(country_name):
    try:
        country_alpha2 = pycountry.countries.get(name=country_name).alpha_2
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except:
        return "Other"
    

# Get full list of country names using pycountry
countries = sorted([country.name for country in pycountry.countries])

# Plotly map generator using dynamic coordinates
def generate_map(country):
    lat, lon = get_country_coordinates(country)
    if lat is None or lon is None:
        lat, lon = 20.0, 0.0 
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


with gr.Blocks() as demo:
    gr.Markdown("## Country Based Subsidiary Report Generator")
    with gr.Row():
        country_dropdown = gr.Dropdown(choices=countries, label="Type or Select a Country")
        generate_btn = gr.Button("Generate Report")
    report_output = gr.Textbox(label="Generated Report")
    map_plot = gr.Plot(label="Country on World Map")
    pdf_download = gr.File(label="Download PDF Report")

    def update_outputs(selected_country):
        summary, pdf_path = generate_report(selected_country)
        map_fig = generate_map(selected_country)
        return summary, map_fig, pdf_path
        #return generate_report(selected_country), generate_map(selected_country)

    generate_btn.click(
        fn=update_outputs,
        inputs=[country_dropdown],
        outputs=[report_output, map_plot, pdf_download]
    )

    #generate_btn.click(fn=update_outputs, inputs=[country_dropdown], outputs=[report_output, map_plot])

demo.launch()
