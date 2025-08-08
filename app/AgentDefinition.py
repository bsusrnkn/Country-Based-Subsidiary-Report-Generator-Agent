# Author: Bengisu Serinken
# August 2025
from typing import Optional
from pydantic import BaseModel, Field
from eventregistry import EventRegistry, QueryArticlesIter, QueryItems
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
import ast, json, requests, pycountry, datetime
import pycountry_convert as pc

# API Keys

gemini_API_key = "REPLACE"
exchange_rate_api_key = "REPLACE"
news_api_key = "REPLACE"    
api_ninja_key = "REPLACE"
    
# LLM Definition

gemini_llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash-lite", 
    google_api_key=gemini_API_key
)

# Data Models
class GDPData(BaseModel):
    country: str
    year: int
    gdp_growth: float
    gdp_nominal: float
    gdp_per_capita_nominal: float
    gdp_ppp: float
    gdp_per_capita_ppp: float
    gdp_ppp_share: float

class ExchangeRateData(BaseModel):
    time_next_update_utc: str
    base_code: str
    conversion_rate_USD: float
    conversion_rate_TRY: float

class NewsArticle(BaseModel):
    title: str
    body: str
    class Config:
        extra = "ignore"
        arbitrary_types_allowed = True   

class UnemploymentData(BaseModel):
    year: int
    unemployment_rate: float
    class Config:
        extra = "ignore"
        arbitrary_types_allowed = True   


class ToolSelectionOutput(BaseModel):
    tool_names: list[str] = Field(..., description="List of tool names to call")

class DataBlob(BaseModel):
    country_name: str = Field(description="Name of the country")
    continent: Optional[str] = Field(default="", description="Continent of the country")
    gdp_data: Optional[GDPData] = None
    currency_code: Optional[str] = Field(default="", description="ISO 4217 currency code for the country")
    exchange_data: Optional[ExchangeRateData] = None
    raw_news_articles: Optional[list[NewsArticle]] = Field(default = [], description="News articles related to the country")
    unemployment_data: Optional[UnemploymentData] = None

class SummaryOutput(BaseModel):
    summary: str = Field(description="Economic summary of the country")

# Parser 

class FlexiblePydanticOutputParser(PydanticOutputParser):
    def parse(self, text: str) -> DataBlob:
        print(f"Parsing text: {text}")
        
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)

        
        stripped = text.strip()
        if stripped.startswith("Final Answer:"):
            stripped = stripped[len("Final Answer:"):].strip()

        # handle double-wrapped JSON: if the model returned a quoted JSON string, unescape it
        if stripped.startswith('"') and stripped.endswith('"'):
            try:
                unwrapped = json.loads(stripped)
                if isinstance(unwrapped, str):
                    stripped = unwrapped
            except json.JSONDecodeError:
                pass
 
        try:
            return super().parse(stripped)
        
        except Exception as e:
            if stripped.count('{') > stripped.count('}'):
                print("Detected missing closing bracket, attempting to fix...")
                stripped_fixed = stripped + "}" * (stripped.count('{') - stripped.count('}'))
                try:
                    return super().parse(stripped_fixed)
                except Exception:
                    pass
            # fallback to Python-literal parsing
            try:
                data = ast.literal_eval(stripped)
            except Exception as lit_err:
                raise OutputParserException(
                    f"Failed to parse as JSON or Python literal: {lit_err}\n\nRaw text:\n{text}"
                )
            try:
                return self.pydantic_object.parse_obj(data)
            except Exception as pd_err:
                raise OutputParserException(
                    f"Pydantic validation failed on fallback data: {pd_err}\n\nParsed data:\n{data}"
                )

# Define parser
parser = FlexiblePydanticOutputParser(pydantic_object=DataBlob)

# Use llm to fix any remaning parsing issues

parser = OutputFixingParser.from_llm(parser=parser, llm=gemini_llm)

# Tools
@tool
def fetch_gdp(country_name: str) -> dict: 
    """Fetches gdp_growth, gdp_nominal, gdp_per_capita_nominal, gdp_ppp, gdp_per_capita_ppp, gdp_ppp_share 
    data including the year of the data for a given country name.
    Input should be a string with the country name."""
    # Frequency: Yearly
    api_url = f'https://api.api-ninjas.com/v1/gdp?country={country_name}'
    response = requests.get(api_url, headers={'X-Api-Key': api_ninja_key})

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

@tool
def fetch_unemployment_rate(country_name: str) -> dict: 
    """Fetches unemployment data for a given country name.
    Input should be a string with the country name."""
    # Frequency: Yearly

    api_url = f"https://api.api-ninjas.com/v1/unemployment?country={country_name}"
    response = requests.get(api_url, headers={'X-Api-Key': api_ninja_key})

    if response.status_code == requests.codes.ok:
        unemployment_data = response.json()
        if unemployment_data:
            data = next((x for x in unemployment_data if x["year"] == datetime.datetime.now().year), None)
            if not data:
                data = max(unemployment_data, key=lambda x: x["year"])  # Fallback to most recent year

            data = max(unemployment_data, key=lambda x: x["year"])
            return {
                "year": data.get("year"),
                "unemployment_rate": data.get("unemployment_rate"),
            }
        else:
            return {"error": "Unemployment data not available."}
        
@tool
def get_continent(country_name: str) -> str:
    """Fetches the continent name of a given country name."""
    try:
        country_alpha2 = pycountry.countries.get(name=country_name).alpha_2
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except:
        return "Other"

@tool
def fetch_currency_code(country_name: str) -> str:
    """Fetches the 3 letter ISO 4217 currency code for a given country name.
    Input should be a string with the country name."""
    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        country_alpha2 = country.alpha_2
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

@tool
def fetch_exchange_rates(currency_given: str) -> dict:
    """Fetches exchange rates from a given country's ISO 4217 currency code to USD and TRY. 
    Also returns the given currency (base_code) code and the next update time of the rates in UTC."""
    # Frequency: Real-time
    currency = currency_given
    if(currency is None):
        return {"error": "Currency code not found for the country."}
    api_url = f"https://v6.exchangerate-api.com/v6/{exchange_rate_api_key}/latest/{currency}"
    response = requests.get(api_url)

    if response:
        data = response.json()
        if data:
            index = 0
            return {
                "time_next_update_utc": data.get("time_next_update_utc"),
                "base_code": data.get("base_code"),
                "conversion_rate_USD": "%.3f" % data.get("conversion_rates", {}).get("USD") if "%.3f" % data.get("conversion_rates", {}).get("USD") != 0 else "%.10f" % data.get("conversion_rates", {}).get("USD"),
                "conversion_rate_TRY": "%.3f" % data.get("conversion_rates", {}).get("TRY") if data.get("conversion_rates", {}).get("TRY") is not None else None
            }
        else:
            return {"error": "Exchange data not available."}
    return 

@tool
def fetch_economic_news(input_data: str) -> list:
    """Fetches recent economic news articles related to a country.
    Input should be a dict (or a string repr of one) with country_name and max_articles which is maximum number of articles, can be set to a number based on the country's development level between 3 - 10.
    """
    if isinstance(input_data, str):
        try:
            input_data = ast.literal_eval(input_data)
        except Exception as e:
            return [f"Invalid input format (could not parse): {e}"]

    if not isinstance(input_data, dict):
        return [f"Invalid input format: expected dict, got {type(input_data).__name__}"]

    try:
        country_name = input_data["country_name"]
        max_articles = int(input_data["max_articles"])
    except Exception as e:
        return [f"Invalid input format: missing or bad 'country_name'/'max_articles': {e}"]

    er = EventRegistry(apiKey=news_api_key)

    q = QueryArticlesIter(
        lang="eng",
        dataType = ["news", "blog"],
        conceptUri = QueryItems.OR([er.getConceptUri("economy"), er.getConceptUri("finance"), 
                                    er.getConceptUri("diplomacy"), er.getConceptUri("investment")]),
        sourceLocationUri=er.getLocationUri(country_name)
    )
    
    raw_articles = list(q.execQuery(er, sortBy="date", maxItems=max_articles))
    # Only keep title and body
    filtered = []
    for art in raw_articles:
        title = getattr(art, "title", None) or art.get("title", "")
        body  = getattr(art, "body",  None) or art.get("body",  "")
        filtered.append({"title": title, "body": body})

    return filtered

# Record tools
tools = [ get_continent,
        fetch_gdp, 
        fetch_currency_code,
        fetch_exchange_rates,
        fetch_economic_news, 
        fetch_unemployment_rate]

# Tool Selector Based on Country Development
chooser = initialize_agent(
    tools = tools,
    llm = gemini_llm,
    agent_type = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True,
    max_iterations = 7,
    enable_thinking = False
)
def tool_selector_agent(country_name: str) -> ToolSelectionOutput:
    
    print(f"Selecting tools for {country_name}...")

    tool_selection_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert economic analyst. 
        Your job is to select the most relevant tools to build an economic report for a given country. 
        Consider the country's economic significance and data availability. 
        For countries with major economies, ALL tools are relevant. 
        For underdeveloped or developing nations, GDP is enough.
        Available tools:
        {tools}
        You don't need to return all the tools, return the ones you think are necessary for the analysis. 
        Return all tools if the country is considered a major economy.
                    
    """),
        ("human", "{question}")
    ])
   
    try:
        structured_llm = gemini_llm.with_structured_output(ToolSelectionOutput)
        print(structured_llm)

        router = tool_selection_prompt | structured_llm

        response = router.invoke({"question": f"Which tools should be used to analyze {country_name}? Provide only the tool names in a list.", "tools" : tools })
        print(f"Tool selection response: {response}")
        
        return response
    except Exception as e:
        raise ValueError(f"Failed to parse tool selection output: {e}")

# Data Collector Based on Selected Tools

def tool_execution_agent(country_name: str, tools_to_run: list[str]) -> DataBlob:
    allowed_tool_str = ", ".join(tools_to_run)
    print(f"Executing tools for {country_name} with allowed tools: {allowed_tool_str}")
    
    format_instructions = parser.get_format_instructions()
    format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")

    prompt = ChatPromptTemplate.from_messages([ 
        ("system", """You are a data collection agent. Your goal is to gather economic data for {country_name}.

        Call the following tools: {allowed_tool_str}. 

        After collecting all required data from given tools, return a single JSON object with exactly these keys:
        - country_name
        - continent
        - gdp_data
        - currency_code
        - exchange_data
        - raw_news_articles
        - unemployment_data

        If there are any missing values, use "", {{}}, or [].

        IMPORTANT:
        - Do NOT include explanations, markdown or commentary.
        - ONLY output the final result as valid JSON.
        - Final Answer: <JSON>
        - When you're done, use `Final Answer:` followed by the JSON object.
        """), 
        ("system", format_instructions), 
        ("human", "Gather data for {country_name} and output only valid JSON.") ])

    selected_tools = [t for t in tools if t.name in tools_to_run]
    executor = initialize_agent(
        tools=selected_tools,
        llm=gemini_llm,
        agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=7,
        enable_thinking=False,
    )
    try:
        prompt = prompt.format_messages(country_name=country_name, allowed_tool_str = allowed_tool_str)     
        response = executor.invoke({"input": prompt})
        text = ""
        if isinstance(response, dict) and "output" in response:
            text = response["output"]
        else:
            text = response if isinstance(response, str) else response.content
        
        data_blob: DataBlob = parser.parse(text)
        return data_blob

    except Exception as e:
        raise ValueError(f"Failed to execute tools: {e}")

# Summary Generator
def summarizer_agent(data_blob: dict, country_name: str) -> SummaryOutput:
    
    print(f"Generating summary for {country_name}...")

    summarizer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an economic reporter. Summarize the relevant news about "
        "the country's situation based on given data, avoid subjective comments be objective, don't mention the articles that do not have any relevant with the country. Focus on news' summary in a minimum 200 words paragraph and return in plain text."),
        ("human", f"Summarize the following data: {data_blob}")
    ])
    
    prompt = summarizer_prompt.format_messages(data_blob = data_blob, country_name = country_name)
    response = gemini_llm.invoke(prompt).content
    
    print(f"Summary response: {response}")
    
    return SummaryOutput(summary=response)

# Run Agent + LLM pipeline
def run_pipeline(country_name: str) -> dict:
    print(f"START OF RUN_PIPELINE for {country_name}")
    
    tools_to_call = tool_selector_agent(country_name).tool_names

    print("Selected tools:", tools_to_call)
    
    data_blob = tool_execution_agent(country_name, tools_to_call)
    
    print("Data collected:", data_blob)

    summary_output = summarizer_agent(data_blob, country_name)
    
    print("Summary generated:", summary_output)
    
    pdf_input = {
        "country_name": data_blob.country_name,
        "gdp_data": data_blob.gdp_data or {},
        "exchange_data": data_blob.exchange_data or {},
        "continent": data_blob.continent or "",
        "summary": summary_output.summary,
        "unemployment_data": data_blob.unemployment_data or {}
    }
    return pdf_input

# Language Translator
def translate_html_to_turkish(html: str) -> str:
    prompt = f"Aşağıdaki HTML içeriğini Türkçeye çevir, HTML yapısını koru:\n\n{html}"
    return gemini_llm.invoke(prompt).content
