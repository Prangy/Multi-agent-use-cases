import functools
import operator
import requests
import os
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

# Set environment variables
os.environ["OPENAI_API_KEY"] = "sk-*****************************"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "laanggraph practice"
os.environ["LANGCHAIN_API_KEY"] = "ls__******************************"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Initialize model
llm = ChatOpenAI(model="gpt-4-turbo-preview")

# Define custom tools
@tool("internet_search", return_direct=False)
def internet_search(query: str) -> str:
    """Searches the internet using DuckDuckGo."""
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
        return results if results else "No results found."

@tool("process_content", return_direct=False)
def process_content(url: str) -> str:
    """Processes content from a webpage."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

tools = [internet_search, process_content]

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

# Define agent node for manager review
def manager_review_node(state):
    messages = state["messages"]
    insight_messages = [msg for msg in messages if msg.name == "Insight_Researcher"]
    if any("No results found" not in msg.content for msg in insight_messages):
        manager_decision = "Manager approves the insights provided by the Insight Researcher. Task complete."
    else:
        manager_decision = "Manager requests the Insight Researcher to provide more detailed insights."
    
    review_text = "Manager Review:\n\n"
    for msg in insight_messages:
        review_text += f"- {msg.content}\n\n"
    
    return {"messages": [HumanMessage(content=review_text + manager_decision, name="Manager")]}

# Create agent node for manager review
manager_review_agent = create_agent(llm, tools, "You are the Manager reviewing the insights.")
manager_review_chain = functools.partial(agent_node, agent=manager_review_agent, name="Manager_Review")

# Set up the workflow
members = ["Web_Searcher", "Insight_Researcher"]
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

search_agent = create_agent(llm, tools, "You are a web searcher. Search the internet for information.")
search_node = functools.partial(agent_node, agent=search_agent, name="Web_Searcher")

insights_research_agent = create_agent(llm, tools, 
        """You are a Insight Researcher. Do step by step. 
        Based on the provided content first identify the list of topics,
        then search internet for each topic one by one
        and finally find insights for each topic one by one.
        Include the insights and sources in the final response
        """)
insights_research_node = functools.partial(agent_node, agent=insights_research_agent, name="Insight_Researcher")

# Define the Agent State, Edges and Graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

workflow = StateGraph(AgentState)
workflow.add_node("Web_Searcher", search_node)
workflow.add_node("Insight_Researcher", insights_research_node)
workflow.add_node("Manager_Review", manager_review_chain)  # Connect Manager_Review node here
workflow.add_node("supervisor", supervisor_chain)

# Define edges
workflow.add_edge("Insight_Researcher", "Manager_Review")  # Connect Insight_Researcher to Manager_Review
# Add an edge from Manager_Review to supervisor
workflow.add_edge("Manager_Review", "supervisor")
for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")

# Compile the graph
graph = workflow.compile()

# Run the graph
for s in graph.stream({
    "messages": [HumanMessage(content="""Search for the latest AI technology trends in 2024,
            summarize the content. After summarise pass it on to insight researcher
            to provide insights for each topic""")]
}):
    if "__end__" not in s:
        print(s)
        print("----")
