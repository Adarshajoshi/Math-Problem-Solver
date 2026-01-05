import streamlit as st
import re
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMChain, LLMMathChain
from langchain_classic.prompts import PromptTemplate
from langchain_classic.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain_classic.utilities import WikipediaAPIWrapper
from langchain_classic.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(
    page_title="Text To Math Problem Solver And Data Search Assistant",
    page_icon="🧮"
)
st.title("Text To Math Problem Solver")

# Groq API key input
groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")
if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# Wikipedia Tool
wikipedia_wrapper = WikipediaAPIWrapper()

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for information on a topic."""
    return wikipedia_wrapper.run(query)

# Helper to extract numeric answer from LLM output
def extract_number(text: str) -> str:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return match.group(0) if match else text

# Math Tool with strict numeric output
math_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a calculator.
Only return the numeric answer to the following expression.
Do not include explanations, text, or extra characters.

Expression: {question}
Answer:
"""
)
math_chain = LLMMathChain.from_llm(llm=llm, prompt=math_prompt)

@tool
def calculator(question: str) -> str:
    """Solve mathematical expressions safely."""
    try:
        result = math_chain.run(question)
        # Extract numeric value to avoid ValueError
        numeric_result = extract_number(result)
        return numeric_result
    except Exception as e:
        return f"Calculator error: {e}"

# Reasoning Tool
reasoning_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an agent tasked with solving users' mathematical questions.
Logically arrive at the solution and provide a detailed explanation in point-wise format.

Question: {question}
Answer:
"""
)
reasoning_chain = LLMChain(llm=llm, prompt=reasoning_prompt)

@tool
def reasoning_tool(question: str) -> str:
    """Answer logic-based or reasoning questions."""
    return reasoning_chain.run(question)

# Initialize agent with all tools
tools = [wikipedia_search, calculator, reasoning_tool]

assistant_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Session state for chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a math chatbot who can answer all your math questions."}
    ]

# Display chat messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# User question input
question = st.text_area(
    "Enter your question:",
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?"
)

# Button to get answer
if st.button("Find My Answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(question, callbacks=[st_cb])

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("### Response:")
            st.success(response)
    else:
        st.warning("Please enter a question.")
