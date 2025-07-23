from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
import requests
from typing import Type
import pycountry
import datetime
import pycountry_convert as pc
import os, subprocess
import descriptions
from weasyprint import HTML
from eventregistry import *

class CountryMetricsInput(BaseModel):
    country_name: str = Field(..., description="The name of the country to generate metrics for.")

class CountryDataTool(BaseTool): 
    name: str = "CountryDataTool"
    description: str = "Fetches country data."
    _llm: any = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True  # Allow non-pydantic types

    def __init__(self, llm=None, **kwargs):
        super().__init__()
        self._llm = llm

    def fetch_gdp(self, country_name: str) -> dict: 
        # Frequency: Yearly

        api_url = f'https://api.api-ninjas.com/v1/gdp?country={country_name}'
        response = requests.get(api_url, headers={'X-Api-Key': '6/hLlVMiS3Fs3DKz7+zk2g==qtyKT1VUfKwmgOba'})
 
        if response.status_code == requests.codes.ok:
            gdp_data = response.json()
            if gdp_data:
                data = next((x for x in gdp_data if x["year"] == datetime.datetime.now().year), None)
                if not data:
                    data = max(gdp_data, key=lambda x: x["year"])  # Fallback to most recent year

                data = max(gdp_data, key=lambda x: x["year"])
                return {
                    "country": country_name,
                    "year": data.get("year"),
                    "gdp_growth": "%.3f" % data.get("gdp_growth") if data.get("gdp_growth") is not None else None,
                    "gdp_nominal": "%.3f" % data.get("gdp_nominal") if data.get("gdp_nominal") is not None else None,
                    "gdp_per_capita_nominal": "%.3f" % data.get("gdp_per_capita_nominal") if data.get("gdp_per_capita_nominal") is not None else None,
                    "gdp_ppp": "%.3f" % data.get("gdp_ppp") if data.get("gdp_ppp") is not None else None,
                    "gdp_per_capita_ppp": "%.3f" % data.get("gdp_per_capita_ppp") if data.get("gdp_per_capita_ppp") is not None else None,
                    "gdp_ppp_share": "%.3f" % data.get("gdp_ppp_share") if data.get("gdp_ppp_share") is not None else None 
                }
            else:
                return {"error": "GDP data not available."}

    def fetch_currency_code(self, country_name: str) -> str:
        # Frequency: Yearly
        try:
            country_alpha2 = pycountry.countries.get(name=country_name).alpha_2
            url = f"https://restcountries.com/v3.1/alpha/{country_alpha2}"
            response = requests.get(url)

            if response.status_code != 200:
                return None

            data = response.json()
            currencies = data[0].get("currencies", {})
            currency_code = list(currencies.keys())[0]  

            return currency_code
        except Exception as e:
            return None

    def fetch_exchange_rates(self, country_name: str) -> dict:
        # Frequency: Real-time
        currency = self.fetch_currency_code(country_name)  # Get the currency code
        if(currency is None):
            return {"error": "Currency code not found for the country."}
        api_key = "b4278a340e112a90d7db9ce0"
        api_url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{currency}"
        response = requests.get(api_url)
 
        if response:
            data = response.json()
            if data:
                return {
                    "time_next_update_utc": data.get("time_next_update_utc"),
                    "base_code": data.get("base_code"),
                    "conversion_rate_USD": "%.3f" % data.get("conversion_rates", {}).get("USD") if "%.3f" % data.get("conversion_rates", {}).get("USD") is not 0 else "%.10f" % data.get("conversion_rates", {}).get("USD"),
                    "conversion_rate_TRY": "%.3f" % data.get("conversion_rates", {}).get("TRY") if data.get("conversion_rates", {}).get("TRY") is not None else None
                }
            else:
                return {"error": "Exchange data not available."}
        return 
    
    def fetch_economic_news(self, country_name: str, max_articles: int) -> any:
        api_key = "bebbfe75-727d-4ed2-a319-bb4989de7208"
        er = EventRegistry(apiKey=api_key)

        q = QueryArticlesIter(
            lang="eng",
            dataType = ["news", "blog"],
            conceptUri = QueryItems.OR([er.getConceptUri("economy"), er.getConceptUri("finance"), 
                                        er.getConceptUri("diplomacy"), er.getConceptUri("investment")]),
            sourceLocationUri=er.getLocationUri(country_name)
        )
        articles = []
        for art in q.execQuery(er, sortBy = "date", maxItems = max_articles):
            articles.append(art)
        return articles

    def summarize_news_with_llm(self, news_data: any) -> str:
        if not news_data:
            return "No major news found."
        news_text = "\n".join([f"{a.get('title', '')}: {a.get('url', '')}" for a in news_data])
        prompt = f"""Summarize the following news headlines and explain their possible impact on the country's economy:\n{news_text}, 
        suppose you are a bank employee generating this report to your C-level manager for him to decide whether to invest or not to 
        the selected country via opening new subsidiaries. Without making any comment, summarize objectively the news by tailoring 
        them such that choosing news that can affect this investment decision. Do not include any information that is not relevant to 
        this decision process. 

        Filter out news not relevant with the given country.
        Only include news that can affect the country's economy, in terms of economic growth, inflation, exchange rates, diplomacy, 
        foreign relations, business environment, investment conditions and risks, finance. 

        After choosing the relevant news as described above, summarize them in a way that is relevant to the report's structure and content.
        
        THE MOST IMPORTANT PART: Use "<h2> News Summary </h2>" in the beginning of the 
        "HTML" format so that I can know where to start printing your result, use it in this format, do not use additional space or newline 
        characters before or after it. Do not use "<h2> News Summary </h2>" anywhere else in your response!

        Avoid any thought processes, subjective comments. Only return "HTML" formatted output using <p> for paragraphs, 
        <ul> for bullet points, <h2> for section titles, <h3> for news item titles. 
        Use this "HTML" format for the report, do not use any other format, do not use any other tags, do not use any other formatting:
        "<h2> News Summary </h2>
        <h3> News Item Title 1 </h3>
        <p>News item description 1</p>
        <h3> News Item Title 2 </h3>
        <p>News item description 2</p>
        etc. "
        Tailor the title of each news item, do not use the original title, tailor the title to make it catchy, relevant to the report's structure and content."""
        return self._llm(prompt).strip()

    def generate_pdf_from_html(self, country_name: str, gdp_data: dict, exchange_data: dict, news_data: str, filename: str = "report.pdf") -> str:
        gdp_items = []
        for key, value in gdp_data.items():
            desc = descriptions.gdp_descriptions.get(key, "") 
            gdp_items.append(f"<li><strong>{desc if desc else key.replace('_', ' ').title()}</strong>: {value}</li>")
        
        exchange_items = []
        for key, value in exchange_data.items():
            desc = descriptions.fx_descriptions.get(key, "") 
            exchange_items.append(f"<li><strong>{desc if desc else key.replace('_', ' ').title()}</strong>: {value}</li>")  

        news_items = self.summarize_news_with_llm(news_data)

        start_keyword = "</think>"
        if start_keyword in news_items:
            start_index = news_items.index(start_keyword) + len(start_keyword)
            news_items = news_items[start_index:] 

        print(news_items)

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
            <h2>GDP Information</h2>
            <ul>
                {''.join(gdp_items)}
            </ul>
            <h2>Exchange Rate Information</h2>
            <ul>
                {''.join(exchange_items)}
            </ul>
            {news_items if news_items else "<p>No major news found.</p>"}
        </body>
        </html>
        """
        filename = f"{country_name}_Subsidiary_Report.pdf"
        HTML(string=html).write_pdf(filename)
        return filename

    def _run(self, country_name: str) -> str:
        try:
            #gdp_data = self.fetch_gdp(country_name)
            exchange_data = self.fetch_exchange_rates(country_name)
            # tex_file = self.generate_latex_report(country_name, gdp_data, exchange_data)
            # self.compile_pdf(tex_file)
            return f"PDF report generated: report.pdf for {country_name}"
        except:
            return "Error fetching country data or generating report."
        
class CountryReportAgent:
    def __init__(self):
        self.llm = Ollama(model="qwen3:1.7b", temperature=0.1)
        self.tool = CountryDataTool(llm=self.llm)
        tools = [self.tool]
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False
        )

    def run(self, country_name: str) -> any:
        #input_data = CountryMetricsInput(country_name=country_name)
        #return self.tool._run(input_data.country_name)
        #return self.agent.run(country_name)
        prompt = f"Generate a detailed economic report in pdf for {country_name}, including GDP, exchange rates, and a summary of recent news that could affect the economy."
        return self.agent.run(prompt)

