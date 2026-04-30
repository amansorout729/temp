from flask import Flask, render_template, request, jsonify
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama
from langchain.tools import DuckDuckGoSearchRun
import numexpr as ne

# Initialize Flask app
app = Flask(__name__)

# ------------------ TOOLS ------------------ #
tools = [
    Tool(
        name="Calculator",
        func=lambda x: str(ne.evaluate(x)),
        description="Evaluates math expressions like '2+2' or '10*5'"
    ),
    Tool(
        name="Search",
        func=DuckDuckGoSearchRun().run,
        description="Search the web for current information"
    ),
]

# ------------------ LLM ------------------ #
llm = Ollama(model="mistral")

# ------------------ MEMORY ------------------ #
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ------------------ AGENT ------------------ #
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# ------------------ ROUTES ------------------ #

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("query")

    try:
        response = agent.run(user_input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)})


# ------------------ RUN ------------------ #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
