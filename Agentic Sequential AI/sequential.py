from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict
from groq import Groq
import os
load_dotenv()
# This below code help to access api key automatically from .env file 
client=Groq()
#print(client)

# This help to   get model of  groq 
llm =ChatGroq(
model="openai/gpt-oss-20b",
temperature=0
)

# Creating state 
class AgentState(TypedDict):
    topic:str
    explanation:str
    summary:str
    question:str


# creating node 
def  generate_explanation(state:AgentState):
    prompt=f"""
    Explain the following topic in simple English
    {state["topic"]}
    """
    response=llm.invoke(prompt)
    return {
        "explanation":response.content
    }

def generate_summary(state:AgentState):
    prompt=f"""
        summarize the given explanation in  3 simple points
        {state["explanation"]}
        """
    response=llm.invoke(prompt)
    return {
        "summary":response.content
    }
def generate_questions(state:AgentState):
    prompt=f"""
    create 5 interview question from given Topic
    {state['summary']}
    """
    response=llm.invoke(prompt)
    return{
        "question":response.content
    }

#  creating the graph
from langgraph.graph import StateGraph,START,END
graph=StateGraph(AgentState)

graph.add_node("explanation",generate_explanation)
graph.add_node("summary",generate_summary)
graph.add_node("question",generate_questions)

# Now connecting the node sequentially

graph.add_edge(START,"explanation")
graph.add_edge("explanation","summary")
graph.add_edge("summary","question")
graph.add_edge("question",END)

app=graph.compile()

result=app.invoke(
    {
    "topic": "Machine Learning",
    "explanation": "",
    "summary": "",
    "question": ""
}
    
)

print(result["explanation"])
print(result["summary"])
print(result["question"])