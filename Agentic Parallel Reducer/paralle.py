from groq import Groq
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from dotenv import load_dotenv
import operator
load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# response=llm.invoke("hello ")
# print(response.content)

class AgentState(TypedDict):
    topic:str
    results:Annotated[list,operator.add]

# creating the node 

def summary_node(state:AgentState):
    prompt=f" Give a short summary of {state['topic']}"
    response=llm.invoke(prompt)

    return{
        "results":["Summary :" + response.content]
    }
def advantage_node(state:AgentState):
    prompt=f"Explain the  main advnatage of {state['topic']}"
    response=llm.invoke(prompt)
    return {
        "results":["Advantages :" + response.content]
    }

def disadvantage_node(state:AgentState):
    prompt=f"Explain Disadvantage of {state['topic']}"
    response=llm.invoke(prompt)
    return{
        "results":["Disadvantage :"+response.content]
    }

# creating the graph

graph=StateGraph(AgentState)

graph.add_node("summary_node",summary_node)
graph.add_node("advantage",advantage_node)
graph.add_node("disadvanatge",disadvantage_node)

graph.add_edge(START,"summary_node")
graph.add_edge(START,"advantage")
graph.add_edge(START,"disadvanatge")

graph.add_edge("summary_node",END)
graph.add_edge("advantage",END)
graph.add_edge("disadvanatge",END)


app=graph.compile()

result=app.invoke(
    {
        "topic":"Mutton Meat",
        "results": []

    }
)

print("\nFINAL RESULTS:\n")

for item in result["results"]:
    print(item)
    print()