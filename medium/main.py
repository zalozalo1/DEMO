from typing import TypedDict, List
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")
# Shared LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, google_api_key=api_key)
# Shared tool
search_tool = DuckDuckGoSearchRun()
# Create two different ReAct agents
travel_agent = create_react_agent(
    model=llm,
    tools=[search_tool],
    prompt="You are a travel assistant that helps users discover places and destinations."
)
planner_agent = create_react_agent(
    model=llm,
    tools=[search_tool],
    prompt="You are a planning assistant that helps users organize their itinerary."
)
# Define LangGraph state
class AgentState(TypedDict):
    messages: List
# Wrap each agent in a node function
def run_travel_agent(state: AgentState) -> AgentState:
    result = travel_agent.invoke(state)
    return {"messages": result["messages"]}
def run_planner_agent(state: AgentState) -> AgentState:
    result = planner_agent.invoke(state)
    return {"messages": result["messages"]}
# Build the graph
graph = StateGraph(AgentState)
# Add nodes
graph.add_node("travel_agent", run_travel_agent)
graph.add_node("planner_agent", run_planner_agent)
# Connect nodes
graph.add_edge(START, "travel_agent")
graph.add_edge("travel_agent", "planner_agent")
graph.add_edge("planner_agent", END)
# Compile
app = graph.compile()
# Run the graph
initial_input = {"messages": [HumanMessage(content="What are the best cities to visit in Spain in summer?")]}
result = app.invoke(initial_input)
# Print the output
for msg in result["messages"]:
    if hasattr(msg, "content"):
        print(f"{msg.type.upper()}: {msg.content}")