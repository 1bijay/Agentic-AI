import streamlit as st
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import operator


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()


# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# -----------------------------
# State
# -----------------------------
class AgentState(TypedDict):
    topic: str
    results: Annotated[list, operator.add]


# -----------------------------
# Nodes
# -----------------------------

def summary_node(state: AgentState):

    prompt = f"Give a short summary of {state['topic']}"

    response = llm.invoke(prompt)

    return {
        "results": [
            "Summary: " + response.content
        ]
    }


def advantage_node(state: AgentState):

    prompt = f"Explain the main advantages of {state['topic']}"

    response = llm.invoke(prompt)

    return {
        "results": [
            "Advantages: " + response.content
        ]
    }


def disadvantage_node(state: AgentState):

    prompt = f"Explain the main disadvantages of {state['topic']}"

    response = llm.invoke(prompt)

    return {
        "results": [
            "Disadvantages: " + response.content
        ]
    }


# -----------------------------
# Create LangGraph
# -----------------------------

graph = StateGraph(AgentState)

graph.add_node("summary_node", summary_node)
graph.add_node("advantage", advantage_node)
graph.add_node("disadvantage", disadvantage_node)


# Parallel branches

graph.add_edge(START, "summary_node")
graph.add_edge(START, "advantage")
graph.add_edge(START, "disadvantage")


# End

graph.add_edge("summary_node", END)
graph.add_edge("advantage", END)
graph.add_edge("disadvantage", END)


app = graph.compile()


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Parallel AI Workflow",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 Parallel AI Workflow")
st.write(
    "This application uses LangGraph to analyze a topic "
    "using three parallel nodes."
)


# Topic input

topic = st.text_input(
    "Enter a topic",
    placeholder="Example: AI"
)


# Run button

if st.button("🚀 Analyze"):

    if not topic:
        st.warning("Please enter a topic.")

    else:

        with st.spinner("Running parallel agents..."):

            result = app.invoke(
                {
                    "topic": topic,
                    "results": []
                }
            )

        st.success("Analysis completed!")


        # -----------------------------
        # Display results
        # -----------------------------

        st.subheader(f"Results for: {topic}")

        results = result["results"]


        # Display each result separately

        for item in results:

            if item.startswith("Summary:"):
                st.markdown("### 📋 Summary")
                st.write(item.replace("Summary:", "", 1).strip())

            elif item.startswith("Advantages:"):
                st.markdown("### ✅ Advantages")
                st.write(item.replace("Advantages:", "", 1).strip())

            elif item.startswith("Disadvantages:"):
                st.markdown("### ⚠️ Disadvantages")
                st.write(item.replace("Disadvantages:", "", 1).strip())