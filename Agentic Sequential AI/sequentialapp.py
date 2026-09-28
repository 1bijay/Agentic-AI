
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# --------------------------------
# Load environment variables
# --------------------------------

load_dotenv()


# --------------------------------
# LLM
# --------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)


# --------------------------------
# State
# --------------------------------

class AgentState(TypedDict):
    topic: str
    explanation: str
    summary: str
    question: str


# --------------------------------
# Node 1: Generate Explanation
# --------------------------------

def generate_explanation(state: AgentState):

    prompt = f"""
    Explain the following topic in simple English:

    {state["topic"]}
    """

    response = llm.invoke(prompt)

    return {
        "explanation": response.content
    }


# --------------------------------
# Node 2: Generate Summary
# --------------------------------

def generate_summary(state: AgentState):

    prompt = f"""
    Summarize the following explanation in 3 simple points:

    {state["explanation"]}
    """

    response = llm.invoke(prompt)

    return {
        "summary": response.content
    }


# --------------------------------
# Node 3: Generate Questions
# --------------------------------

def generate_questions(state: AgentState):

    prompt = f"""
    Create 5 interview questions from the following topic:

    {state["topic"]}

    Summary:
    {state["summary"]}
    """

    response = llm.invoke(prompt)

    return {
        "question": response.content
    }


# --------------------------------
# Create LangGraph
# --------------------------------

graph = StateGraph(AgentState)

graph.add_node("explanation", generate_explanation)
graph.add_node("summary", generate_summary)
graph.add_node("question", generate_questions)

# Sequential flow
graph.add_edge(START, "explanation")
graph.add_edge("explanation", "summary")
graph.add_edge("summary", "question")
graph.add_edge("question", END)

app = graph.compile()


# =================================
# Streamlit UI
# =================================

st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Learning Assistant")
st.write("Learn any topic using a sequential LangGraph workflow.")


# Topic input
topic = st.text_input(
    "Enter a topic",
    placeholder="Example: Machine Learning"
)


# Generate button
if st.button("Generate", type="primary"):

    if not topic:
        st.warning("Please enter a topic.")

    else:

        with st.spinner("AI is processing your topic..."):

            result = app.invoke(
                {
                    "topic": topic,
                    "explanation": "",
                    "summary": "",
                    "question": ""
                }
            )

        # -------------------------
        # Explanation
        # -------------------------

        st.subheader("📚 Explanation")

        st.write(result["explanation"])


        # -------------------------
        # Summary
        # -------------------------

        st.subheader("📝 Summary")

        st.write(result["summary"])


        # -------------------------
        # Interview Questions
        # -------------------------

        st.subheader("🎯 Interview Questions")

        st.write(result["question"])

