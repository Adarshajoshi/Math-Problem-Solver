Text To Math Problem Solver And Data Search Assistant 🧮

A Streamlit-based AI assistant that can solve math problems, answer reasoning questions, and fetch Wikipedia data in an interactive chat interface. Powered by LangChain Classic and Google Gemma 2 (Groq API).

Features

Mathematical Problem Solving
Solve arithmetic, algebra, powers, roots, and complex word problems with numeric parsing and LLM reasoning.

Logic & Reasoning
Step-by-step solutions for logic puzzles and reasoning-based questions.

Wikipedia Integration
Search Wikipedia for any topic and return summarized results.

Interactive Chat Interface
Streamlit-based chat interface with conversation history for smooth user interaction.


Installation

Clone the repository:

git clone https://github.com/Adarshajoshi/Math-Problem-Solver
cd Math-Problem-Solver


Create a virtual environment:

python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py

Configuration

Groq API Key:
Enter your Groq API key in the Streamlit sidebar to initialize the LLM.
