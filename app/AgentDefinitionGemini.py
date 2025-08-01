from typing import Optional, List, Dict
from pydantic import BaseModel, Field
import ast, json, re, requests, pycountry, datetime
import pycountry_convert as pc
from eventregistry import EventRegistry, QueryArticlesIter, QueryItems
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

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

class ToolSelectionOutput(BaseModel):
    tool_names: list[str] = Field(..., description="List of tool names to call")

class DataBlob(BaseModel):
    country_name: str = Field(description="Name of the country")
    continent: Optional[str] = Field(default="", description="Continent of the country")
    gdp_data: Optional[GDPData] = None
    currency_code: Optional[str] = Field(default="", description="ISO 4217 currency code for the country")
    exchange_data: Optional[ExchangeRateData] = None
    raw_news_articles: Optional[list[NewsArticle]] = Field(default = [], description="News articles related to the country")

#parser = PydanticOutputParser(pydantic_object=DataBlob)

class FlexiblePydanticOutputParser(PydanticOutputParser):
    def parse(self, text: str) -> DataBlob:
        print(f"Parsing text: {text}")
        # strip off markdown fences
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)

        # strip any “Final Answer:” prefix
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
 
         # first, try the normal JSON-based parse
        try:
            return super().parse(stripped)
        
        ##########################
        except Exception:
            # fallback to Python-literal parsing
            try:
                data = ast.literal_eval(stripped)
            except Exception as lit_err:
                raise OutputParserException(
                    f"Failed to parse as JSON or Python literal: {lit_err}\n\nRaw text:\n{text}"
                )
            # now validate via Pydantic
            try:
                return self.pydantic_object.parse_obj(data)
            except Exception as pd_err:
                raise OutputParserException(
                    f"Pydantic validation failed on fallback data: {pd_err}\n\nParsed data:\n{data}"
                )


parser = FlexiblePydanticOutputParser(pydantic_object=DataBlob)

class SummaryOutput(BaseModel):
    summary: str = Field(description="Economic summary of the country")

# LLM Definitions

ollama_llm = ChatOllama(model="qwen3:4b", temperature=0.1)

gemini_API_key = "AIzaSyDj6X2bAElb4WHWYrfrK0lrjo8Syp3FLMI"
gemini_llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash-lite",  # or "models/gemini-pro" or "models/gemini-1.0-pro"
    google_api_key=gemini_API_key
)

# Tools
@tool
def fetch_gdp(country_name: str) -> dict: 
    """Fetches gdp_growth, gdp_nominal, gdp_per_capita_nominal, gdp_ppp, gdp_per_capita_ppp, gdp_ppp_share 
    data including the year of the data for a given country name.
    Input should be a string with the country name."""
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
    api_key = "b4278a340e112a90d7db9ce0"
    api_url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{currency}"
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

    api_key = "bebbfe75-727d-4ed2-a319-bb4989de7208"
    er = EventRegistry(apiKey=api_key)

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
        fetch_economic_news]

# Agent 1: Tool Selector Based on Country Development
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
        ("system", """You are an expert economic analyst. Your job is to select the most relevant tools to build an economic report for a given country. 
        Consider the country's economic significance and data availability. For major economies, all tools are relevant. For smaller or less-developed nations, core data like GDP and currency might be sufficient.
        Available tools:
                    {tools}
                    You don't need to return all the tools, just the ones you think are necessary for the analysis.
                    
    """),
        ("human", "{question}")
    ])
    print("111")
    
    # prompt = tool_selection_prompt.format_messages(country_name=country_name)
    # print("prompt:", prompt)
    
    try:
        structured_llm = gemini_llm.with_structured_output(ToolSelectionOutput)
        print(structured_llm)

        router = tool_selection_prompt | structured_llm

        response = router.invoke({"question": f"Which tools should be used to analyze {country_name}? Provide only the tool names in a list.", "tools" : tools })
        print(f"Tool selection response: {response}")
        
        return response
    except Exception as e:
        raise ValueError(f"Failed to parse tool selection output: {e}")

# Agent 2: Data Collector Based on Selected Tools
executor = initialize_agent(
    tools = tools,
    llm = gemini_llm,
    agent_type = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True,
    max_iterations = 7,
    enable_thinking = False
)
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

        If missing, use "", {{}}, or [].

        IMPORTANT:
        - Do NOT include explanations, markdown or commentary.
        - ONLY output the final result as valid JSON.
        - Final Answer: <JSON>
        - When you're done, use `Final Answer:` followed by the JSON object.
        """), 
        ("system", format_instructions), 
        ("human", "Gather data for {country_name} and output only valid JSON.") ])

    exprompt = """
You are a data collection agent. Your goal is to gather economic data for {country_name}.

Call the following tools: {allowed_tool_str}. 

After collecting all required data from given tools, return a single JSON object with exactly these keys:
- country_name
- continent
- gdp_data
- currency_code
- exchange_data
- raw_news_articles

If missing, use "", {{}}, or [].

IMPORTANT:
- Do NOT include explanations, markdown or commentary.
- ONLY output the final result as valid JSON.
- Final Answer: <JSON>
- When you're done, use `Final Answer:` followed by the JSON object.
"""

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

# Agent 3: Summary Generator

def summarizer_agent(data_blob: dict, country_name: str) -> SummaryOutput:
    
    print(f"Generating summary for {country_name}...")

    summarizer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an economic reporter. Filter out irrelevent news given and summarize the relevant news about "
        "the country's economic situation based on given data, avoid subjective comments be objective, in min 100 words and return in plain text."),
        ("human", f"Summarize the following data: {data_blob}")
    ])
    
    prompt = summarizer_prompt.format_messages(data_blob = data_blob, country_name = country_name)
    response = gemini_llm.invoke(prompt).content
    
    print(f"Summary response: {response}")
    
    return SummaryOutput(summary=response)
    
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
        "summary": summary_output.summary
    }
    return pdf_input

# @tool
# def process_news_articles(news_data: any) -> str:
#     """Processes raw news articles into a clean HTML format.
#     Input should be a dict with country_name and articles keys.
#     Returns HTML string with news articles or a 'no news' message.
#     """
#     if isinstance(news_data, str):
#         try:
#             news_data = ast.literal_eval(news_data)
#         except:
#             return "<p>Error processing news data</p>"

#     if not isinstance(news_data, dict):
#         return "<p>Invalid news data format</p>"

#     country_name = news_data.get("country_name", "")
#     articles = news_data.get("articles", [])
    
#     if not articles:
#         return f"<h2>News Summary</h2><p>No news articles found for {country_name}.</p>"

#     relevant_articles = []
#     for article in articles:
#         # More robust relevance check
#         title = article.get('title', '').lower()
#         body = article.get('body', '').lower()
#         if (country_name.lower() in title or 
#             country_name.lower() in body or
#             any(word in title or word in body 
#                 for word in [country_name.split()[0].lower(), 
#                            f"the {country_name.lower()}"])):
#             relevant_articles.append(article)

#     if not relevant_articles:
#         return f"<h2>News Summary</h2><p>No relevant news found for {country_name}.</p>"

#     # Generate HTML for relevant articles
#     html_articles = []
#     for article in relevant_articles[:3]:  # Limit to 3 most relevant
#         date = article.get('dateTimePub', '').split('T')[0] if article.get('dateTimePub') else 'Unknown date'
#         html_articles.append(f"""
#         <div class="news-article">
#             <h3>{article.get('title', 'No title')}</h3>
#             <p><strong>Source:</strong> {article.get('source', {}).get('title', 'Unknown')} | <strong>Date:</strong> {date}</p>
#             <p>{article.get('body', 'No content available.')[:300]}...</p>
#             <p><a href="{article.get('url', '#')}">Read more</a></p>
#         </div>
#         <hr>
#         """)

#     return f"""
#     <h2>News Summary</h2>
#     <p>Relevant economic news for {country_name}:</p>
#     {''.join(html_articles)}
#     """


# def data_collecting_agent_invoking_function(country_name: str) -> dict:
#     """Invokes the agent with the given input and returns the PDF filename result."""
    
# #     prompt = f"""You are an intelligent economic analyst agent. Your goal is to evaluate whether {country_name} is developed, developing, or underdeveloped based on your general knowledge and any available information.

# #         Based on your assessment:
# #         - Dynamically choose which tools to call from the available list.
# #         - For example:
# #             - Use minimal tools for underdeveloped countries
# #             - Use all tools for developed countries
    
# #         After calling the tools, return a single dictionary with:
# #         - `"country_name"` (string)
# #         - `"continent"` (str)
# #         - `"gdp_data"` (dict or None)
# #         - `"exchange_data"` (dict or None)
# #         - `"news_data"` (HTML string summary)
# #         - `"summary"` (a concise plain text in HTML summarizing economic outlook)

# #         If any tool fails, include empty values but preserve the keys.

# #         Final output format (return as JSON, no markdown):

# #         {{
# #         "country_name": "...",
# #         "continent": "...",
# #         "gdp_data": {{ ... }},
# #         "exchange_data": {{ ... }},
# #         "news_data": "...",
# #         "summary": "..."
# #         }}
# # """
#     data_collection_prompt = f"""
#     You are creating a comprehensive economic report for {country_name}. Follow these steps carefully:

#     1. Gather Core Data:
#     - Call `get_continent` for "{country_name}"
#     - Call `fetch_gdp` for "{country_name}"
#     - Call `fetch_currency_code` for "{country_name}"

#     2. Get Exchange Rates:
#     - Use the currency code from Step 1 (fetch_currency_code) to call `fetch_exchange_rates`
#     - If no currency code found, use "USD" as fallback

#     3. Process News:
#     - Call `fetch_economic_news` with {{"country_name": "{country_name}", "max_articles": 10}}
#     - Then Call `process_news_articles` with {{"country_name": "{country_name}", "articles": [...]}} # article HTML string

#     Return EXACTLY this dictionary:
#     {{
#         "country_name": "{country_name}",
#         "gdp_data": {{...}},  # from fetch_gdp
#         "exchange_data": {{...}},  # from fetch_exchange_rates
#         "news_data": ...,  # return value of process_news_articles
#         "continent": "..."  # from get_continent
#     }}

#     You must call only the relevant tools and return this dictionary.
#     """

#     try:
#         #Collecting data
#         data_blob: dict = agent.run(data_collection_prompt.format(country_name=country_name))
#         #parsed_result = json.loads(data_blob)
#         print(f"Data collected: {data_blob}")
    
#         return data_blob
    
#     except Exception as e:
#         # Create a more detailed error PDF
#         error_html = f"""
#         <html>
#         <body>
#             <h1>Error Generating Report for {country_name}</h1>
#             <p><strong>Error:</strong> {str(e)}</p>
#         </body>
#         </html>
#         """
#         fallback_filename = f"{country_name}_Error_Report.pdf"
#         HTML(string=error_html).write_pdf(fallback_filename)
#         return os.path.abspath(fallback_filename)

# def summarizing_agent_invoking_function(data_blob: dict) -> str:
#     """Invokes the agent with the given input and returns the summary result."""
    
#     summary_prompt = f"""You are an economic analyst.

#     Summarize the economic situation of the country using the following data:

#     {data_blob}

#     Focus on:
#     - Currency stability
#     - Economic growth
#     - Key trends from the news
#     - Opportunities and risks

#     Output an analysis (min 100 words) in plain text. No markdown. Return a str.
#     """

#     try:
#         #Summarizing the data
#         summary: str = agent.invoke(summary_prompt.format(data_blob=data_blob))
#         return summary
    
#     except Exception as e:
#         # Create a more detailed error PDF
#         error_html = f"""
#         <html>
#         <body>
#             <h1>Error Generating Report</h1>
#             <p><strong>Error:</strong> {str(e)}</p>
#         </body>
#         </html>
#         """
#         fallback_filename = f"Error_Report.pdf"
#         HTML(string=error_html).write_pdf(fallback_filename)
#         return os.path.abspath(fallback_filename)


# def agent_invoking_function(country_name: str) -> str:
#     """Invokes the agent with the given input and returns the PDF filename result."""  
#     prompt = f"""Your mission is to create a single PDF economic report for {country_name}. You must strictly follow every step.

#         - Call `get_continent` with the country name. If it returns "Other", use your knowledge to determine the correct continent.
#         - Call `fetch_gdp` with the country name. If it fails, use empty values but note this in the report.
#         - Call `fetch_currency_code` with the country name. 
#         - Using the currency code of fetch_currency_code's output, call `fetch_exchange_rates`.

#         - Call `fetch_economic_news` with dict with keys country_name and max_articles (set to 10).
#         - Save the output as a list of articles.
#         - Call `process_news_articles` with dict with keys country_name and articles (output of fetch_economic_news). 

#         - Collect all gathered data.
#         - Action: Call `generate_pdf_from_html` with dict with keys country_name, gdp_data (output of fetch_gdp), 
#                                                             exchange_data (output of fetch_exchange_rates), 
#                                                             news_data (output of process_news_articles), 
#                                                             continent (output of get_continent).

#         - After calling `generate_pdf_from_html`, extract the filename it returns.
#         - Do NOT include any additional commentary, explanation, markdown, or formatting.
#         - Your final response must EXACTLY be the ouptut of generate_pdf_from_html (no asterisks or markdown).

#         - Use the EXACT dictionary structure shown above for generate_pdf_from_html.
#         You must always respond in the following format:

#         Thought: [your internal reasoning]
#         Action: [name of a tool from the available tool list]
#         Action Input: [a JSON dictionary that matches the tool's schema]

#         When you are ready to give the final result, say:

#         Final Answer: [the filename string returned by generate_pdf_from_html]

#         Do not return markdown, bullet points, or formatted summaries.
#         Never respond outside this structure. Strictly follow this format.
#     """ 
#     prompt_template = PromptTemplate.from_template(prompt)
#     formatted_prompt = prompt_template.format(country_name=country_name)

#     result = agent.run({"input": formatted_prompt})

#     filename = result.strip()
#     print(result)

#     return os.path.join(os.getcwd(), filename)


# Better result def agent_invoking_function(country_name: str) -> str:
#     """Invokes the agent with the given input and returns the PDF filename result."""
    
#     prompt = f"""You are creating a comprehensive economic report for {country_name}. Follow these steps carefully:

# 1. Gather Core Data:
#    - Call `get_continent` for "{country_name}"
#    - Call `fetch_gdp` for "{country_name}"
#    - Call `fetch_currency_code` for "{country_name}"

# 2. Get Exchange Rates:
#    - Use the currency code from Step 1 to call `fetch_exchange_rates`
#    - If no currency code found, use USD as fallback

# 3. Process News:
#    - Call `fetch_economic_news` with {{"country_name": "{country_name}", "max_articles": 10}}
#    - Call `process_news_articles` with {{"country_name": "{country_name}", "articles": [news_articles]}}

# 4. Generate PDF:
#    - Collect all gathered data
#    - Call `generate_pdf_from_html` with this EXACT structure (no changes to keys):
#    {{
#        "country_name": "{country_name}",
#        "gdp_data": {{...}},  # from fetch_gdp
#        "exchange_data": {{...}},  # from fetch_exchange_rates
#        "news_data": "...",  # return value of process_news_articles
#        "continent": "..."  # from get_continent
#    }}

# 5. Final Answer:
#    - Return ONLY the filename return value of generate_pdf_from_html
#    - Format: "Final Answer: filename.pdf"

# Important Rules:
# - Never skip steps
# - Use exact dictionary keys as shown
# - If any step fails, use empty values but include all required keys
# - Verify data before including in PDF
# """

#     try:
#         result = agent.run(prompt)
        
#         # Clean up the filename
#         filename = result.replace("Final Answer:", "").strip()
#         if filename.endswith(".pdf"):
#             filename = f"{country_name}_Subsidiary_Report.pdf"

#         # Verify the PDF was created
#         if not os.path.exists(filename):
#             raise FileNotFoundError(f"PDF file {filename} was not created")

#         return os.path.abspath(filename)
    
#     except Exception as e:
#         # Create a more detailed error PDF
#         error_html = f"""
#         <html>
#         <body>
#             <h1>Error Generating Report for {country_name}</h1>
#             <p><strong>Error:</strong> {str(e)}</p>
#         </body>
#         </html>
#         """
#         fallback_filename = f"{country_name}_Error_Report.pdf"
#         HTML(string=error_html).write_pdf(fallback_filename)
#         return os.path.abspath(fallback_filename)