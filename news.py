import functools, operator, requests, os, json
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from newsapi import NewsApiClient

# Set environment variables
# Initialize model with your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "laanggraph practice"
os.environ["LANGCHAIN_API_KEY"] = "your key"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Initialize model
llm = ChatOpenAI(model="gpt-4-turbo-preview")

# Initialize News API client
newsapi = NewsApiClient(api_key='a53bb8bbce60438180b59626237e4278')

#define custom tool
@tool("news_gathering", return_direct=False)
def news_gathering(query: str) -> str:
    """Gathers news articles using the News API."""
    # Make a request to News API
    response = newsapi.get_everything(q=query, language='en', sort_by='publishedAt', page_size=5)
    articles = response['articles']
    headlines = [article['title'] for article in articles]
    return headlines if headlines else "No news found."

@tool("process_content", return_direct=False)
def process_content(url: str) -> str:
    """Processes content from a webpage."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

tools = [news_gathering, process_content]

# 2. Agents 
# Helper function for creating agents
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# Define agent nodes
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# Create Agent Supervisor
members = ["News_gatherer", "Editor"]
system_prompt = (
    "As a supervisor, your role is to oversee a dialogue between these"
    " workers: {members}. Based on the user's request,"
    " determine which worker should take the next action. Each worker is responsible for"
    " executing a specific task and reporting back their findings and progress. Once all tasks are complete,"
    " indicate with 'FINISH'."
)

options = ["FINISH"] + members
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}] }},
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
]).partial(options=str(options), members=", ".join(members))

supervisor_chain = (prompt | llm.bind_functions(functions=[function_def], function_call="route") | JsonOutputFunctionsParser())

# Define agents
news_gathering_agent = create_agent(llm, tools, "You are a News Gathering Agent. Gather news from the News API.")
news_gathering_node = functools.partial(agent_node, agent=news_gathering_agent, name="News_gatherer")

edit_news_agent = create_agent(llm, tools,
                               """You are an Edit News Agent. 
    Your task is to edit news articles based on the user's request. 
    Process the provided news content and make necessary edits.""")

edit_news_node = functools.partial(agent_node, agent=edit_news_agent, name="Editor")

# Define the Agent State, Edges and Graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

workflow = StateGraph(AgentState)
workflow.add_node("News_gatherer", news_gathering_node)
workflow.add_node("Editor", edit_news_node)
workflow.add_node("supervisor", supervisor_chain)

# Define edges
for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")

graph = workflow.compile()

# Run the graph
for s in graph.stream({
    "messages": [HumanMessage(content="""Please find news articles about latest news 
                             and summarize it . After summarization pass 
                              it on to Edit news agent""")]
}, config={"recursion_limit": 50}):  # Set a higher recursion limit
    if "__end__" not in s:
        print(s)
        print("----")

# Run the graph
final_response = graph.invoke({
     "messages": [HumanMessage(
         content="""Please find news articles about latest  news 
                              and summarize it . After summarization pass 
                              it on to Edit news agent""")]
})

