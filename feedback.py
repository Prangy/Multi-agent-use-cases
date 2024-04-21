import functools
import operator
import os
from typing import Annotated, Sequence, TypedDict
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# Set environment variables
# Initialize model with your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "customer_feedback_analysis"
os.environ["LANGCHAIN_API_KEY"] = "your key"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Initialize model
llm = ChatOpenAI(model="gpt-4-turbo-preview")

# Define custom tools
@tool("collect_feedback", return_direct=False)
def collect_feedback(query: str) -> str:
    """Collects feedback from various sources."""
    # Placeholder implementation
    return "Feedback collected: " + query  

@tool("analyze_sentiment", return_direct=False)
def analyze_sentiment(feedback: str) -> str:
    """Analyzes sentiment of collected feedback."""
    # Placeholder implementation
    return "Sentiment analyzed for: " + feedback  

@tool("identify_topic", return_direct=False)
def identify_topic(feedback: str) -> str:
    """Identifies key topics from analyzed feedback."""
    # Placeholder implementation
    return "Topics identified for: " + feedback  

@tool("generate_insight", return_direct=False)
def generate_insight(topics: str) -> str:
    """Generates insights from identified topics."""
    # Placeholder implementation
    return "Insights generated for topics: " + topics  

tools = [collect_feedback, analyze_sentiment, identify_topic, generate_insight]

# Define agents
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# Define agent states, edges, and graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

workflow = StateGraph(AgentState)
workflow.add_node("Feedback_Collector", functools.partial(agent_node, agent=create_agent(llm, tools, "You are a Feedback Collector. Collect feedback from various sources."), name="Feedback_Collector"))
workflow.add_node("Sentiment_Analyzer", functools.partial(agent_node, agent=create_agent(llm, tools, "You are a Sentiment Analyzer. Analyze sentiment of collected feedback."), name="Sentiment_Analyzer"))
workflow.add_node("Topic_Identifier", functools.partial(agent_node, agent=create_agent(llm, tools, "You are a Topic Identifier. Identify key topics from analyzed feedback."), name="Topic_Identifier"))
workflow.add_node("Insight_Generator", functools.partial(agent_node, agent=create_agent(llm, tools, "You are an Insight Generator. Generate insights from identified topics."), name="Insight_Generator"))

members = ["Feedback_Collector", "Sentiment_Analyzer", "Topic_Identifier", "Insight_Generator"]
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

workflow.add_node("supervisor", supervisor_chain)

for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")

graph = workflow.compile()

# Run the graph with the feedback query
feedback_query = "The product has some good features, but there are areas that could be improved."

for s in graph.stream({"messages": [HumanMessage(content=feedback_query)]}):
    if "__end__" not in s:
        print(s)
        print("----")

final_response = graph.invoke({"messages": [HumanMessage(content=feedback_query)]})
print(final_response['messages'][1].content)
